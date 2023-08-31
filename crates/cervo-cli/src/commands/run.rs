use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::prelude::{Inferer, InfererExt, Response, State};
use clap::Parser;

use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

/// Run a model once.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct Args {
    /// The model file to use - ONNX, NNEF or CRVO format.
    file: PathBuf,

    /// The batch size to feed the network.
    #[clap(short, long)]
    batch_size: usize,

    /// An epsilon key to randomize noise.
    #[clap(short, long)]
    with_epsilon: Option<String>,

    #[clap(long)]
    print_output: bool,

    #[clap(long)]
    print_input: bool,
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

fn indent_by(target: String, prefix_len: usize) -> String {
    let prefix = " ".repeat(prefix_len);

    target
        .lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn print_input(obs: &HashMap<u64, State<'_>>) {
    let formatted = format!("{:#?}", obs);
    let indented = indent_by(formatted, 4);
    println!("Inputs:\n{}", indented);
}

fn print_output(obs: &HashMap<u64, Response<'_>>) {
    let formatted = format!("{:#?}", obs);
    let indented = indent_by(formatted, 4);
    println!("Outputs:\n{}", indented);
}

pub(super) fn run(config: Args) -> Result<()> {
    let mut reader = File::open(&config.file)?;
    let inferer = if cervo::nnef::is_nnef_tar(&config.file) {
        cervo::nnef::builder(&mut reader).build_fixed(&[config.batch_size])?
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => cervo::onnx::builder(&mut reader).build_fixed(&[config.batch_size])?,
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

        if config.print_input {
            print_input(&observations);
        }

        inferer.infer_batch(observations.clone())?;

        let start = Instant::now();
        let res = inferer.infer_batch(observations)?;

        let dur = start.elapsed();
        if config.print_output {
            print_output(&res);
        }

        dur
    } else {
        let shapes = inferer.input_shapes().to_vec();
        let observations = build_inputs_from_desc(config.batch_size as u64, &shapes);
        inferer.infer_batch(observations.clone())?;

        let start = Instant::now();
        let res = inferer.infer_batch(observations)?;

        let dur = start.elapsed();
        if config.print_output {
            print_output(&res);
        }

        dur
    };

    println!(
        "Ran in {:.2} ms, {:.2} ms per element ({} elements), ",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / config.batch_size as f64,
        config.batch_size,
    );

    Ok(())
}
