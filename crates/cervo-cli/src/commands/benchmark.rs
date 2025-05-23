use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::prelude::{Batcher, Inferer, InfererExt, State};
use clap::Parser;
use clap::ValueEnum;
use serde::Serialize;
use std::{collections::HashMap, fs::File, path::PathBuf, time::Instant};

fn number_range_parser(num: &str) -> Result<Vec<usize>, String> {
    let mut nums = vec![];

    for segment in num.split(',') {
        if segment.contains('-') {
            let mut parts = segment.split('-');
            let lower: usize = parts
                .next()
                .ok_or_else(|| "no lower end".to_owned())?
                .parse()
                .map_err(|e| format!("failed parsing number: {e:?}"))?;
            let upper: usize = parts
                .next()
                .ok_or_else(|| "no lower end".to_owned())?
                .parse()
                .map_err(|e| format!("failed parsing number: {e:?}"))?;

            for i in lower..=upper {
                nums.push(i);
            }
        } else {
            let value: usize = segment
                .parse()
                .map_err(|e| format!("failed parsing number: {e:?}"))?;
            nums.push(value);
        }
    }

    Ok(nums)
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

    /// Output format: text or json.
    #[clap(long, value_enum, default_value = "text")]
    output: OutputFormat,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
enum OutputFormat {
    Text,
    Json,
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

#[derive(Serialize, Clone)]
struct Record {
    batch_size: usize,
    mean: f64,
    stddev: f64,
    total: f64,
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
        total: m * batch_size as f64,
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
    let mut records: Vec<Record> = Vec::new();
    for batch_size in config.batch_sizes {
        let mut reader = File::open(&config.file)?;
        let mut inferer = if cervo::nnef::is_nnef_tar(&config.file) {
            cervo::nnef::builder(&mut reader).build_fixed(&[batch_size])?
        } else {
            match config.file.extension().and_then(|ext| ext.to_str()) {
                Some("onnx") => cervo::onnx::builder(&mut reader).build_fixed(&[batch_size])?,
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
                .filter(|(k, _)| k.as_str() != epsilon)
                .cloned()
                .collect::<Vec<_>>();

            let observations = build_inputs_from_desc(batch_size as u64, &shapes);

            execute_load_metrics(batch_size, observations, config.count, &mut inferer)?
        } else {
            let shapes = inferer.input_shapes().to_vec();
            let observations = build_inputs_from_desc(batch_size as u64, &shapes);

            execute_load_metrics(batch_size, observations, config.count, &mut inferer)?
        };
        records.push(record.clone());

        // Print Text
        if matches!(config.output, OutputFormat::Text) {
            println!(
                "Batch Size {}: {:.2} ms ± {:.2} per element, {:.2} ms total",
                record.batch_size, record.mean, record.stddev, record.total,
            );
        }
    }
    // Print JSON
    if matches!(config.output, OutputFormat::Json) {
        let json = serde_json::to_string_pretty(&records)?;
        println!("{}", json);
    }
    Ok(())
}
