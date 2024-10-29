use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::inferer::{InfererBuilder, InfererProvider};
use cervo::core::prelude::{Inferer, InfererExt, Response, State};
use clap::Parser;
use std::io::Read;
use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex};
use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

use std::collections::VecDeque;
mod request_generated;
mod response_generated;
mod types_generated;

pub struct Semaphore {
    condvar: Condvar,
    queue: Mutex<usize>,
}

pub struct Permit<'a> {
    pool: &'a Semaphore,
}

pub struct OwnedPermit {
    pool: Arc<Semaphore>,
}

impl OwnedPermit {
    pub fn acquire(pool: Arc<Semaphore>) -> Self {
        std::mem::forget(pool.acquire());

        OwnedPermit { pool }
    }

    pub fn try_acquire(pool: Arc<Semaphore>) -> Option<Self> {
        let maybe_permit = pool.try_acquire();

        if maybe_permit.is_some() {
            std::mem::forget(maybe_permit);
            Some(OwnedPermit { pool })
        } else {
            None
        }
    }
}

impl Drop for OwnedPermit {
    fn drop(&mut self) {
        self.pool.release();
    }
}

impl Semaphore {
    pub fn new(initial: usize) -> Self {
        Semaphore {
            queue: Mutex::new(initial),
            condvar: Condvar::new(),
        }
    }

    pub fn try_acquire(&self) -> Option<Permit<'_>> {
        let mut guard = self.queue.lock().unwrap();
        if *guard == 0 {
            return None;
        }

        *guard -= 1;

        Some(Permit { pool: self })
    }

    pub fn acquire(&self) -> Permit<'_> {
        let mut item = self
            .condvar
            .wait_while(self.queue.lock().unwrap(), |counter| *counter == 0)
            .unwrap();

        *item -= 1;

        Permit { pool: self }
    }

    pub fn release(&self) {
        let mut guard = self.queue.lock().unwrap();
        *guard += 1;
        self.condvar.notify_one();
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.pool.release();
    }
}

#[derive(Clone, Copy, Parser, Debug, Default)]
pub enum InfererMode {
    Simple,
    Fixed,
    #[default]
    Memoizing,
}

impl FromStr for InfererMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simple" => Ok(InfererMode::Simple),
            "fixed" => Ok(InfererMode::Fixed),
            "memoizing" => Ok(InfererMode::Memoizing),
            _ => Err(format!("unknown inferer mode: {}", s)),
        }
    }
}

impl InfererMode {
    fn from_model<I: InfererProvider>(
        self,
        model: InfererBuilder<I>,
        maybe_batch_size: Option<usize>,
    ) -> Result<Box<dyn Inferer>> {
        let boxed: Box<dyn Inferer> = match self {
            InfererMode::Simple => Box::new(model.build_basic()?),
            InfererMode::Fixed => Box::new(model.build_fixed(&[
                maybe_batch_size.expect("must provide batch size when using a fixed batcher"),
            ])?),
            InfererMode::Memoizing => Box::new(model.build_memoizing(&[])?),
        };

        Ok(boxed)
    }
}
/// Run a model once.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct Args {
    /// The model file to use - ONNX, NNEF or CRVO format.
    file: PathBuf,

    /// An epsilon key to randomize noise.
    #[clap(short, long)]
    with_epsilon: Option<String>,

    #[clap(short, long)]
    inferer_mode: InfererMode,

    #[clap(short, long, default_value = None)]
    maybe_batch_size: Option<usize>,

    #[clap(short, long, default_value = "11223")]
    port: u16,

    #[clap(long, default_value = "0.0.0.0")]
    host: String,
}

pub(super) fn serve(config: Args) -> Result<()> {
    let mut reader = File::open(&config.file)?;
    let inferer = if cervo::nnef::is_nnef_tar(&config.file) {
        config
            .inferer_mode
            .from_model(cervo::nnef::builder(&mut reader), config.maybe_batch_size)?
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => config
                .inferer_mode
                .from_model(cervo::onnx::builder(&mut reader), config.maybe_batch_size)?,
            Some("crvo") => config.inferer_mode.from_model(
                InfererBuilder::new(AssetData::deserialize(&mut reader)?),
                config.maybe_batch_size,
            )?,
            Some(other) => bail!("unknown file type {:?}", other),
            None => bail!("missing file extension {:?}", config.file),
        }
    };

    serve_inner(inferer, config.host, config.port)
}

fn serve_inner(model: Box<dyn Inferer>, host: String, port: u16) -> Result<()> {
    use tiny_http::{Response, Server};

    // The requests are expected to be in the form of a POST request with the body containing the input data.
    // The encoding of the input data is expected to be in the following format:
    // 1. The first byte is the batch size.
    // For each batch item:
    //   2. The first byte is the number of inputs
    //   For each input:
    //     3. The first byte is the length of the input key
    //     4. The next `length` bytes are the input key as utf-8
    //     5. The next 4 bytes is the byte-length of the input value as LE unsigned
    //     6. The next `length` bytes are the input value as float in LE

    fn run_server(
        server: Arc<Server>,
        tx: std::sync::mpsc::Sender<(tiny_http::Request, Vec<u8>)>,
        model: Arc<Box<dyn Inferer>>,
        semaphore: Arc<Semaphore>,
    ) -> Result<()> {
        let mut batch = cervo::core::batcher::Batcher::new(model.as_ref());
        let mut responders: Vec<(tiny_http::Request, Vec<u64>)> = vec![];
        loop {
            match server.recv_timeout(std::time::Duration::from_millis(5)) {
                Err(e) => {
                    eprintln!("Error: {:?}", e);
                }
                Ok(None) => {
                    if batch.is_empty() {
                        continue;
                    }

                    let Some(permit) = OwnedPermit::try_acquire(semaphore.clone()) else {
                        continue;
                    };

                    let responders = std::mem::take(&mut responders);
                    let mut batch = std::mem::replace(
                        &mut batch,
                        cervo::core::batcher::Batcher::new(model.as_ref()),
                    );
                    let model = model.clone();

                    let semaphore = semaphore.clone();
                    std::thread::spawn(move || {
                        let mut result = batch.execute(model.as_ref()).unwrap();

                        for (request, ids) in responders {
                            let mut buf = vec![];
                            buf.push(ids.len() as u8);
                            for id in ids {
                                let response = result.remove(&(id as u64)).unwrap();
                                buf.push(response.data.len() as u8);
                                for (key, value) in response.data {
                                    buf.push(key.len() as u8);
                                    buf.extend(key.as_bytes());
                                    buf.extend(&(value.len() as u32 * 4).to_le_bytes());
                                    buf.extend(value.iter().flat_map(|f| f.to_le_bytes()));
                                }
                            }
                            request.respond(Response::from_data(buf.clone()))?;
                        }

                        drop(permit);
                        Ok::<_, anyhow::Error>(())
                    });
                }
                Ok(Some(mut request)) => {
                    let mut buf = vec![];
                    request.as_reader().read_to_end(&mut buf)?;
                    let mut offset = 0;
                    let batch_size = buf[offset];
                    offset += 1;

                    let mut copy_data = Vec::<f32>::new();

                    let mut responder_ids = vec![];
                    for id in 0..batch_size {
                        let input_count = buf[offset];
                        offset += 1;

                        let mut state = State::empty();
                        for _ in 0..input_count {
                            let key_length = buf[offset] as usize;
                            offset += 1;

                            let key = std::str::from_utf8(&buf[offset..offset + key_length])?;

                            offset += key_length;

                            let value_length = u32::from_le_bytes([
                                buf[offset],
                                buf[offset + 1],
                                buf[offset + 2],
                                buf[offset + 3],
                            ]) as usize
                                / 4;

                            offset += 4;

                            let values_part = unsafe {
                                let ptr = buf.as_ptr().offset(offset as isize);
                                ptr as *const f32
                            };

                            let values =
                                unsafe { std::slice::from_raw_parts(values_part, value_length) };
                            copy_data.extend_from_slice(values);
                            state.data.insert(key, values.to_vec());
                            offset += value_length * 4;
                        }

                        let id = batch.len() as u64;
                        batch.push(id, state)?;
                        responder_ids.push(id);
                    }
                    responders.push((request, responder_ids));

                    if batch.len() >= 40 {
                        let mut result = batch.execute(model.as_ref()).unwrap();

                        for (request, ids) in responders.drain(..) {
                            let mut buf = vec![];
                            buf.push(ids.len() as u8);
                            for id in ids {
                                let response = result.remove(&(id as u64)).unwrap();
                                buf.push(response.data.len() as u8);
                                for (key, value) in response.data {
                                    buf.push(key.len() as u8);
                                    buf.extend(key.as_bytes());
                                    buf.extend(&(value.len() as u32 * 4).to_le_bytes());
                                    buf.extend(value.iter().flat_map(|f| f.to_le_bytes()));
                                }
                            }
                            request.respond(Response::from_data(buf.clone()))?;
                        }
                    }
                }
            }
        }
        Ok::<_, anyhow::Error>(())
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let mut model = Arc::new(model);
    let addr = format!("{}:{}", host, port);
    let server = Server::http(addr).unwrap();

    let semaphore = Arc::new(Semaphore::new(12));

    let server = Arc::new(server);

    for _ in 0..6 {
        let server = server.clone();
        let tx = tx.clone();
        let model = model.clone();
        let semaphore = semaphore.clone();
        std::thread::spawn(move || {
            run_server(server, tx, model, semaphore).unwrap();
        });
    }

    run_server(server, tx, model, semaphore).unwrap();

    Ok(())
}
