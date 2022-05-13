// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 12 May 2022

/*!

*/

use anyhow::Result;
use std::{
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use structopt::StructOpt;
use tractor_onnx::simple_inferer_from_stream;
#[derive(Debug, StructOpt)]
pub(crate) struct LoadComparison {
    #[structopt(long = "onnx", short = "o")]
    onnx: Option<PathBuf>,
    #[structopt(long = "nnef", short = "n")]
    nnef: Option<PathBuf>,

    iterations: usize,
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
    format: String,
    kind: String,
    mean: f64,
    stddev: f64,
}
fn execute_load_metrics<T: Fn(&mut dyn Read) -> Result<()>>(
    format: &str,
    kind: &str,
    file: &Path,
    count: usize,
    load_fn: T,
) -> Result<Record> {
    let data = std::fs::read(file)?;
    let mut times = vec![];

    for _ in 0..count {
        let mut cursor = Cursor::new(&data);
        let start = Instant::now();
        black_box(&(load_fn(&mut cursor)?));
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let (m, s) = (mean(&times).unwrap(), std_deviation(&times).unwrap());

    Ok(Record {
        format: format.to_owned(),
        kind: kind.to_owned(),
        mean: m,
        stddev: s,
    })
}

#[inline(never)]
fn check_onnx_simple(o: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("onnx", "simple", o, iterations, |read| {
        simple_inferer_from_stream(read)?;
        Ok(())
    })
}

#[inline(never)]
fn check_nnef_simple(n: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("nnef", "simple", n, iterations, |read| {
        tractor_nnef::simple_inferer_from_stream(read)?;
        Ok(())
    })
}

#[inline(never)]
fn check_onnx_dynamic(o: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("onnx", "dynamic", o, iterations, |read| {
        tractor_onnx::batched_inferer_from_stream(read, &[])?;
        Ok(())
    })
}

#[inline(never)]
fn check_nnef_dynamic(n: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("nnef", "dynamic", n, iterations, |read| {
        tractor_nnef::batched_inferer_from_stream(read, &[])?;
        Ok(())
    })
}

#[inline(never)]
fn check_onnx_fixed(o: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("onnx", "fixed", o, iterations, |read| {
        tractor_onnx::fixed_batch_inferer_from_stream(read, &[1, 2, 4])?;
        Ok(())
    })
}

#[inline(never)]
fn check_nnef_fixed(n: &Path, iterations: usize) -> Result<Record> {
    execute_load_metrics("nnef", "fixed", n, iterations, |read| {
        tractor_nnef::fixed_batch_inferer_from_stream(read, &[1, 2, 4])?;
        Ok(())
    })
}

pub(crate) fn compare_loadtimes(config: LoadComparison) -> Result<()> {
    let mut records = if let Some(o) = config.onnx.as_ref() {
        vec![
            check_onnx_fixed(o, config.iterations)?,
            check_onnx_dynamic(o, config.iterations)?,
            check_onnx_simple(o, config.iterations)?,
        ]
    } else {
        vec![]
    };

    let r = if let Some(n) = config.nnef.as_ref() {
        vec![
            check_nnef_fixed(n, config.iterations)?,
            check_nnef_dynamic(n, config.iterations)?,
            check_nnef_simple(n, config.iterations)?,
        ]
    } else {
        vec![]
    };

    records.extend(r);

    let mut file = std::fs::File::create(config.output_file)?;
    for record in records {
        writeln!(
            file,
            "{},{},{},{}",
            record.format, record.kind, record.mean, record.stddev
        )?;
    }
    Ok(())
}
