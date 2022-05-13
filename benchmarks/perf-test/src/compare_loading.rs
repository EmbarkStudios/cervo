// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 12 May 2022

/*!

*/

use anyhow::Result;
use std::{
    io::{Cursor, Read},
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

fn execute_load_metrics<T: Fn(&mut dyn Read) -> Result<()>>(
    format: &str,
    kind: &str,
    file: &Path,
    load_fn: T,
) -> Result<()> {
    let data = std::fs::read(file)?;
    let mut times = vec![];

    for _ in 0..100 {
        let mut cursor = Cursor::new(&data);
        let start = Instant::now();
        black_box(&(load_fn(&mut cursor)?));
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let (m, s) = (mean(&times).unwrap(), std_deviation(&times).unwrap());
    eprintln!("{},{},{:6.2},{:6.2}", format, kind, m, s);

    Ok(())
}

#[inline(never)]
fn check_onnx_simple(o: &Path) {
    execute_load_metrics("onnx", "simple", o, |read| {
        simple_inferer_from_stream(read)?;
        Ok(())
    })
    .unwrap();
}

#[inline(never)]
fn check_nnef_simple(n: &Path) {
    execute_load_metrics("nnef", "simple", n, |read| {
        tractor_nnef::simple_inferer_from_stream(read)?;
        Ok(())
    })
    .unwrap();
}

#[inline(never)]
fn check_onnx_dynamic(o: &Path) {
    execute_load_metrics("onnx", "dynamic", o, |read| {
        tractor_onnx::batched_inferer_from_stream(read, &[])?;
        Ok(())
    })
    .unwrap();
}

#[inline(never)]
fn check_nnef_dynamic(n: &Path) {
    execute_load_metrics("nnef", "dynamic", n, |read| {
        tractor_nnef::batched_inferer_from_stream(read, &[])?;
        Ok(())
    })
    .unwrap();
}

#[inline(never)]
fn check_onnx_fixed(o: &Path) {
    execute_load_metrics("onnx", "fixed", o, |read| {
        tractor_onnx::fixed_batch_inferer_from_stream(read, &[1, 2, 4])?;
        Ok(())
    })
    .unwrap();
}

#[inline(never)]
fn check_nnef_fixed(n: &Path) {
    execute_load_metrics("nnef", "fixed", n, |read| {
        tractor_nnef::fixed_batch_inferer_from_stream(read, &[1, 2, 4])?;
        Ok(())
    })
    .unwrap();
}

pub(crate) fn compare_loadtimes(config: LoadComparison) {
    if let Some(o) = config.onnx.as_ref() {
        check_onnx_fixed(o);
        check_onnx_dynamic(o);
        check_onnx_simple(o);
    }

    if let Some(n) = config.nnef.as_ref() {
        check_nnef_fixed(n);
        check_nnef_dynamic(n);
        check_nnef_simple(n);
    }
}
