use anyhow::{bail, Result};
use cervo_asset::AssetData;
use cervo_core::prelude::{Batcher, Inferer, InfererExt, State};
use clap::Parser;
use number_range::NumberRangeOptions;
use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

fn number_range_parser(num: &str) -> Result<Vec<usize>, String> {
    let rng_parser = NumberRangeOptions::default();
    let rng = rng_parser.with_range_sep('-').parse::<usize>(num)?;
    Ok(rng.collect())
}

/// Run a model with different batch sizes to estimate performance.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct Args {
    /// The model file to use - ONNX, NNEF or CRVO format.
    file: PathBuf,

    /// The batch sizes to test.
    #[clap(
        short,
        long,
        value_name = "N1-N2,N3",
        value_parser(number_range_parser)
    )]
    batch_sizes: std::vec::Vec<usize>,

    /// How many total elements to test per batch-size.
    #[clap(short, long, default_value = "1000")]
    count: usize,

    /// An epsilon key to randomize noise.
    #[clap(short, long)]
    with_epsilon: Option<String>,
}

fn mean(data: &[f64]) -> Option<f64> {
    let sum = data.iter().sum::<f64>();
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f64),
        _ => None,
    }
}

fn std_deviation(data: &[f64]) -> Option<f64> {
    let data_mean = mean(data)?;
    let count = data.len();

    let variance = data
        .iter()
        .map(|value| {
            let diff = data_mean - value;

            diff * diff
        })
        .sum::<f64>()
        / count as f64;

    Some(variance.sqrt())
}

fn black_box<T>(dummy: T) -> T {
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::read_volatile(&dummy)
    }
}

struct Record {
    batch_size: usize,
    mean: f64,
    stddev: f64,
}
fn execute_load_metrics<I: Inferer>(
    batch_size: usize,
    data: HashMap<u64, State<'_>>,
    count: usize,
    inferer: &mut I,
) -> Result<Record> {
    let mut times = vec![];

    let mut batcher = Batcher::new(inferer);
    for _ in 0..10 {
        let batch = data.clone();
        batcher.extend(batch)?;
        black_box(&(batcher.execute(inferer)?));
    }

    let mut batcher = Batcher::new(inferer);
    for _ in 0..(count / batch_size) {
        let start = Instant::now();
        let batch = data.clone();
        batcher.extend(batch)?;
        black_box(&(batcher.execute(inferer)?));
        times.push(start.elapsed().as_secs_f64() * 1000.0 / batch_size as f64);
    }

    let (m, s) = (mean(&times).unwrap(), std_deviation(&times).unwrap());

    Ok(Record {
        batch_size,
        mean: m,
        stddev: s,
    })
}

pub fn build_inputs_from_desc(
    count: u64,
    inputs: &[(String, Vec<usize>)],
) -> HashMap<u64, State<'_>> {
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
    for batch_size in config.batch_sizes {
        let mut reader = File::open(&config.file)?;
        let mut inferer = if cervo_nnef::is_nnef_tar(&config.file) {
            cervo_nnef::builder(&mut reader).build_fixed(&[batch_size])?
        } else {
            match config.file.extension().and_then(|ext| ext.to_str()) {
                Some("onnx") => cervo_onnx::builder(&mut reader).build_fixed(&[batch_size])?,
                Some("crvo") => AssetData::deserialize(&mut reader)?.load_fixed(&[batch_size])?,
                Some(other) => bail!("unknown file type {:?}", other),
                None => bail!("missing file extension {:?}", config.file),
            }
        };

        let record = if let Some(epsilon) = config.with_epsilon.as_ref() {
            let mut inferer = inferer.with_default_epsilon(epsilon)?;
            // TODO[TSolberg]: Issue #31.
            let shapes = inferer
                .input_shapes()
                .iter()
                .cloned()
                .filter(|(k, _)| k.as_str() != epsilon)
                .collect::<Vec<_>>();

            let observations = build_inputs_from_desc(batch_size as u64, &shapes);

            execute_load_metrics(batch_size, observations, config.count, &mut inferer)?
        } else {
            let shapes = inferer.input_shapes().to_vec();
            let observations = build_inputs_from_desc(batch_size as u64, &shapes);

            execute_load_metrics(batch_size, observations, config.count, &mut inferer)?
        };

        println!(
            "Batch Size {}: {:.2} ms ± {:.2} per element, {:.2} ms total",
            record.batch_size,
            record.mean,
            record.stddev,
            record.mean * record.batch_size as f64,
        );
    }
    Ok(())
}
