use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::inferer::{InfererBuilder, InfererProvider};
use cervo::core::prelude::{Inferer, State};
use clap::Parser;
use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex};
use std::{fs::File, path::PathBuf};

mod request_capnp;

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

    #[clap(short, long, default_value = "8")]
    threads: u16,

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

    serve_inner(inferer, config.host, config.port, config.threads)
}

fn serve_inner(model: Box<dyn Inferer>, host: String, port: u16, threads: u16) -> Result<()> {
    use tiny_http::{Response, Server};

    fn run_server(
        server: Arc<Server>,
        _tx: std::sync::mpsc::Sender<(tiny_http::Request, Vec<u8>)>,
        model: Arc<Box<dyn Inferer>>,
        semaphore: Arc<Semaphore>,
    ) -> Result<()> {
        let mut batch = cervo::core::batcher::Batcher::new(model.as_ref());
        let mut responders: Vec<(tiny_http::Request, Vec<u64>)> = vec![];
        loop {
            match server.try_recv() {
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
                            let mut message = capnp::message::Builder::new_default();
                            {
                                let response =
                                    message.init_root::<request_capnp::response::Builder<'_>>();

                                let mut data_instances = response.init_data(ids.len() as _);
                                for (idx, id) in ids.into_iter().enumerate() {
                                    let mut instance = data_instances.reborrow().get(idx as _);
                                    instance.set_identity(id as _);
                                    let response = result.remove(&(id as u64)).unwrap();

                                    let mut dls =
                                        instance.init_data_lists(response.data.len() as _);

                                    for (index, (key, value)) in
                                        response.data.into_iter().enumerate()
                                    {
                                        let mut data_list = dls.reborrow().get(index as _);
                                        data_list.set_name(key);
                                        data_list.set_values(&value[..])?;
                                    }
                                }
                            };
                            let data = message.get_segments_for_output();
                            let mut buf = vec![];

                            match data {
                                capnp::OutputSegments::SingleSegment(d) => {
                                    buf.extend(d[0]);
                                }
                                capnp::OutputSegments::MultiSegment(s) => {
                                    for d in s {
                                        buf.extend(d);
                                    }
                                }
                            }

                            let response = Response::from_data(buf);
                            request.respond(response)?;
                        }

                        drop(permit);
                        Ok::<_, anyhow::Error>(())
                    });
                }
                Ok(Some(mut request)) => {
                    use capnp::serialize_packed;

                    let mut buf = vec![];

                    request.as_reader().read_to_end(&mut buf)?;
                    let reader =
                        serialize_packed::read_message(&buf[..], Default::default()).unwrap();
                    let data = reader
                        .get_root::<'_, request_capnp::request::Reader<'_>>()
                        .unwrap();

                    let mut responder_ids = vec![];

                    for instance in data.get_data().unwrap() {
                        let mut state = State::empty();
                        for datalist in instance.get_data_lists().unwrap() {
                            let input = datalist.get_values()?;
                            let key = datalist.get_name()?;

                            state
                                .data
                                .insert(key.to_str()?, input.as_slice().unwrap().to_vec());
                        }

                        let id = batch.len() as u64;
                        batch.push(id, state)?;
                        responder_ids.push(id);
                    }
                    responders.push((request, responder_ids));

                    if batch.len() >= 12 {
                        let mut result = batch.execute(model.as_ref()).unwrap();

                        for (request, ids) in responders.drain(..) {
                            let mut message = capnp::message::Builder::new_default();
                            {
                                let response =
                                    message.init_root::<request_capnp::response::Builder<'_>>();

                                let mut data_instances = response.init_data(ids.len() as _);
                                for (idx, id) in ids.into_iter().enumerate() {
                                    let mut instance = data_instances.reborrow().get(idx as _);
                                    instance.set_identity(id as _);
                                    let response = result.remove(&(id as u64)).unwrap();

                                    let mut dls =
                                        instance.init_data_lists(response.data.len() as _);

                                    for (index, (key, value)) in
                                        response.data.into_iter().enumerate()
                                    {
                                        let mut data_list = dls.reborrow().get(index as _);
                                        data_list.set_name(key);
                                        data_list.set_values(&value[..])?;
                                    }
                                }
                            };
                            let data = message.get_segments_for_output();
                            buf.clear();

                            match data {
                                capnp::OutputSegments::SingleSegment(d) => {
                                    buf.extend(d[0]);
                                }
                                capnp::OutputSegments::MultiSegment(s) => {
                                    for d in s {
                                        buf.extend(d);
                                    }
                                }
                            }

                            let response = Response::from_data(&buf[..]);
                            request.respond(response)?;
                        }
                    }
                }
            }
        }
    }

    let (tx, _rx) = std::sync::mpsc::channel();
    let model = Arc::new(model);
    let addr = format!("{}:{}", host, port);
    let server = Server::http(addr).unwrap();

    let semaphore = Arc::new(Semaphore::new(12));

    let server = Arc::new(server);

    for _ in 0..threads {
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
