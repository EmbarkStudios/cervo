// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

/*!

*/

use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::Result;
use structopt::StructOpt;
use tractor::{EpsilonInjector, Inferer, LowQualityNoiseGenerator};
use tractor_onnx::fixed_batch_inferer_from_stream;

fn black_box<T>(dummy: T) -> T {
    unsafe { std::ptr::read_volatile(&dummy) }
}

#[derive(StructOpt, Debug)]
pub(crate) struct NoiseComparison {
    steps: usize,
    batch_size: usize,
    output_file: PathBuf,
}

struct Measurement {
    kind: &'static str,
    step: usize,
    time: Duration,
}

fn execute_steps(
    mut inferer: impl Inferer,
    kind: &'static str,
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let observations =
        crate::helpers::build_inputs_from_desc(batch_size as u64, inferer.input_shapes());

    let mut measurements = vec![];
    for step in 0..steps {
        let obs = observations.clone();
        let start = Instant::now();
        let res = inferer.infer(obs)?;
        black_box(&res);
        let elapsed = start.elapsed();

        measurements.push(Measurement {
            kind,
            step,
            time: elapsed,
        });
    }

    Ok(measurements)
}

fn test_hq(onnx: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let raw = fixed_batch_inferer_from_stream(&mut reader, &[batch_size])?;
    let instance = EpsilonInjector::wrap(raw, "epsilon")?;

    execute_steps(instance, "low", steps, batch_size)
}

fn test_lq(onnx: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let raw = fixed_batch_inferer_from_stream(&mut reader, &[batch_size])?;
    let instance =
        EpsilonInjector::with_generator(raw, LowQualityNoiseGenerator::default(), "epsilon")?;

    execute_steps(instance, "high", steps, batch_size)
}

pub(crate) fn execute_comparison(onnx_path: &Path, config: NoiseComparison) -> Result<()> {
    let lq = test_lq(onnx_path, config.steps, config.batch_size)?;
    let hq = test_hq(onnx_path, config.steps, config.batch_size)?;

    let mut file = std::fs::File::create(config.output_file)?;

    let denom = config.batch_size as f64;
    for series in [lq, hq] {
        for row in series {
            writeln!(
                file,
                "{:?},{},{}",
                row.kind,
                row.step,
                row.time.as_secs_f64() * 1e6 / denom
            )?;
        }
    }

    Ok(())
}
