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
use cervo_core::prelude::{Inferer, InfererExt, LowQualityNoiseGenerator};
use clap::Parser;

fn black_box<T>(dummy: T) -> T {
    unsafe { std::ptr::read_volatile(&dummy) }
}

#[derive(Parser, Debug)]
pub(crate) struct NoiseComparison {
    input_file: PathBuf,
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
    let inputs = inferer.input_shapes().to_vec();
    let observations = crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs);

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

    let instance = cervo_onnx::builder(&mut reader)
        .build_fixed(&[batch_size])?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "low", steps, batch_size)
}

fn test_lq(onnx: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let instance = cervo_onnx::builder(&mut reader)
        .build_fixed(&[batch_size])?
        .with_epsilon(LowQualityNoiseGenerator::default(), "epsilon")?;

    execute_steps(instance, "high", steps, batch_size)
}

pub(crate) fn execute_comparison(config: NoiseComparison) -> Result<()> {
    let onnx_path = &config.input_file;
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
