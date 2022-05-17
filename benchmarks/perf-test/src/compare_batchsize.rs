// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 12 May 2022

/*!

*/

use anyhow::Result;
use cervo_core::{Inferer, State};
use std::{
    collections::HashMap,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use cervo_onnx::{
    batched_inferer_from_stream, direct_inferer_from_stream, fixed_batch_inferer_from_stream,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
pub(crate) struct BatchScaling {
    #[structopt(long = "onnx", short = "o")]
    onnx: PathBuf,

    iterations: usize,

    #[structopt(short = "b", use_delimiter = true)]
    batch_sizes: Vec<usize>,

    output_file: PathBuf,
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
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
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
        _ => None,
    }
}

fn black_box<T>(dummy: T) -> T {
    unsafe { std::ptr::read_volatile(&dummy) }
}

struct Record {
    kind: &'static str,
    batch_size: usize,
    mean: f64,
    stddev: f64,
}
fn execute_load_metrics<I: Inferer>(
    kind: &'static str,
    batch_size: usize,
    data: HashMap<u64, State>,
    count: usize,
    inferer: &mut I,
) -> Result<Record> {
    let mut times = vec![];

    for _ in 0..10 {
        let batch = data.clone();
        black_box(&(inferer.infer(batch)?));
    }

    for _ in 0..(count / batch_size) {
        let batch = data.clone();
        let start = Instant::now();
        black_box(&(inferer.infer(batch)?));
        times.push(start.elapsed().as_secs_f64() * 1000.0 / batch_size as f64);
    }

    let (m, s) = (mean(&times).unwrap(), std_deviation(&times).unwrap());

    Ok(Record {
        kind,
        batch_size,
        mean: m,
        stddev: s,
    })
}

#[inline(never)]
fn run_batch_size(o: &Path, batch_sizes: Vec<usize>, iterations: usize) -> Result<Vec<Record>> {
    std::io::stdout().flush().unwrap();
    let data = std::fs::read(o)?;

    let mut records = vec![];

    records.extend(
        batch_sizes
            .clone()
            .into_iter()
            .map(|batch_size| {
                println!("Checking batch size: {:?}", batch_size);

                let mut inferer =
                    fixed_batch_inferer_from_stream(&mut Cursor::new(&data), &[batch_size])?;
                let batch = crate::helpers::build_inputs_from_desc(
                    batch_size as u64,
                    inferer.input_shapes(),
                );

                execute_load_metrics("fixed", batch_size, batch, iterations, &mut inferer)
            })
            .collect::<Result<Vec<_>>>()?,
    );

    records.extend(
        batch_sizes
            .clone()
            .into_iter()
            .map(|batch_size| {
                println!("Checking batch size: {:?}", batch_size);

                let mut inferer = cervo_onnx::simple_inferer_from_stream(&mut Cursor::new(&data))?;
                let batch = crate::helpers::build_inputs_from_desc(
                    batch_size as u64,
                    inferer.input_shapes(),
                );

                execute_load_metrics("single", batch_size, batch, iterations, &mut inferer)
            })
            .collect::<Result<Vec<_>>>()?,
    );

    records.extend(
        batch_sizes
            .clone()
            .into_iter()
            .map(|batch_size| {
                println!("Checking batch size: {:?}", batch_size);

                let mut inferer =
                    batched_inferer_from_stream(&mut Cursor::new(&data), &[batch_size])?;
                let batch = crate::helpers::build_inputs_from_desc(
                    batch_size as u64,
                    inferer.input_shapes(),
                );

                execute_load_metrics("dynamic", batch_size, batch, iterations, &mut inferer)
            })
            .collect::<Result<Vec<_>>>()?,
    );

    records.extend(
        batch_sizes
            .into_iter()
            .map(|batch_size| {
                println!("Checking batch size: {:?}", batch_size);

                let mut inferer = direct_inferer_from_stream(&mut Cursor::new(&data))?;
                let batch = crate::helpers::build_inputs_from_desc(
                    batch_size as u64,
                    inferer.input_shapes(),
                );

                execute_load_metrics("direct", batch_size, batch, iterations, &mut inferer)
            })
            .collect::<Result<Vec<_>>>()?,
    );

    Ok(records)
}

pub(crate) fn compare_batch_scaling(config: BatchScaling) -> Result<()> {
    let records = run_batch_size(&config.onnx, config.batch_sizes, config.iterations)?;

    let mut file = std::fs::File::create(config.output_file)?;
    for record in records {
        writeln!(
            file,
            "{},{},{},{}",
            record.kind, record.batch_size, record.mean, record.stddev
        )?;
    }
    Ok(())
}
