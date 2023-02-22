use anyhow::{bail, Result};
use cervo_asset::AssetData;
use cervo_core::prelude::{Inferer, InfererExt, State};
use clap::Parser;

use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

/// Shortly describe the model file.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct Args {
    file: PathBuf,
    #[clap(short, long)]
    batch_size: usize,

    #[clap(short, long)]
    with_epsilon: Option<String>,
}

fn build_inputs_from_desc(count: u64, inputs: &[(String, Vec<usize>)]) -> HashMap<u64, State<'_>> {
    (0..count)
        .map(|idx| {
            (
                idx,
                State {
                    data: inputs
                        .iter()
                        .map(|(key, count)| {
                            (key.as_str(), vec![idx as f32; count.iter().product()])
                        })
                        .collect(),
                },
            )
        })
        .collect()
}

pub(super) fn run(config: Args) -> Result<()> {
    let mut reader = File::open(&config.file)?;
    let inferer = if cervo_nnef::is_nnef_tar(&config.file) {
        cervo_nnef::builder(&mut reader).build_fixed(&[config.batch_size])?
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => cervo_onnx::builder(&mut reader).build_fixed(&[config.batch_size])?,
            Some("crvo") => {
                AssetData::deserialize(&mut reader)?.load_fixed(&[config.batch_size])?
            }
            Some(other) => bail!("unknown file type {:?}", other),
            None => bail!("missing file extension {:?}", config.file),
        }
    };

    let elapsed = if let Some(epsilon) = config.with_epsilon.as_ref() {
        let inferer = inferer.with_default_epsilon(epsilon)?;
        // TODO[TSolberg]: Issue #31.
        let shapes = inferer
            .input_shapes()
            .iter()
            .cloned()
            .filter(|(k, _)| k.as_str() != epsilon)
            .collect::<Vec<_>>();

        let observations = build_inputs_from_desc(config.batch_size as u64, &shapes);
        inferer.infer_batch(observations.clone())?;

        let start = Instant::now();
        inferer.infer_batch(observations)?;
        start.elapsed()
    } else {
        let shapes = inferer.input_shapes().to_vec();
        let observations = build_inputs_from_desc(config.batch_size as u64, &shapes);
        inferer.infer_batch(observations.clone())?;

        let start = Instant::now();
        inferer.infer_batch(observations)?;
        start.elapsed()
    };

    println!(
        "Ran in {:.2} ms, {:.2} ms per element ({} elements), ",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / config.batch_size as f64,
        config.batch_size,
    );

    Ok(())
}
