use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::inferer::{InfererBuilder, InfererProvider};
use cervo::core::prelude::{Inferer, InfererExt, Response, State};
use clap::Parser;
use std::io::Read;
use std::str::FromStr;
use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

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

    let mut model = model.into_batched();
    let addr = format!("{}:{}", host, port);
    let server = Server::http(addr).unwrap();

    let mut buf = vec![];
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

    for mut request in server.incoming_requests() {
        buf.clear();
        request.as_reader().read_to_end(&mut buf)?;

        let mut offset = 0;
        let batch_size = buf[offset];
        offset += 1;

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
                ]) as usize;

                offset += 4;

                let values_part = unsafe { buf.as_ptr().offset(offset as isize) as *const f32 };
                let values = unsafe { std::slice::from_raw_parts(values_part, value_length) };

                state.data.insert(key, values.to_vec());
                offset += value_length * 4;
            }

            model.push(id as u64, state)?;
        }

        let mut result = model.execute()?;

        buf.clear();
        buf.push(result.len() as u8);

        for id in 0..result.len() {
            let response = result.remove(&(id as u64)).unwrap();
            buf.push(response.data.len() as u8);

            for (key, value) in response.data {
                buf.push(key.len() as u8);
                buf.extend(key.as_bytes());
                buf.extend(&(value.len() as u32).to_le_bytes());
                buf.extend(value.iter().flat_map(|f| f.to_le_bytes()));
            }
        }

        request.respond(Response::from_data(buf.clone()))?;
    }

    Ok(())
}
