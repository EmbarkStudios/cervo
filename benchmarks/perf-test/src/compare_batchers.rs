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
use tractor::{EpsilonInjector, Inferer};
use tractor_onnx::{
    batched_inferer_from_stream, fixed_batch_inferer_from_stream, simple_inferer_from_stream,
};

fn black_box<T>(dummy: T) -> T {
    unsafe { std::ptr::read_volatile(&dummy) }
}

#[derive(StructOpt, Debug)]
pub(crate) struct BatcherComparison {
    #[structopt(short = "-f", long = "--fixed", use_delimiter = true)]
    fixed_sizes: Vec<usize>,
    #[structopt(short = "-d", long = "--dynamic", use_delimiter = true)]
    dynamic_sizes: Vec<usize>,
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
    perchance::seed_global(0xff00ff00ff00ff00ff00ff00ff00ff00u128);
    let observations =
        crate::helpers::build_inputs_from_desc(batch_size as u64, inferer.input_shapes());

    let mut measurements = vec![];
    for step in 0..steps {
        let obs = if batch_size > 0 {
            observations.clone()
        } else {
            let batch_size = perchance::global().uniform_range_usize(1..10);
            crate::helpers::build_inputs_from_desc(batch_size as u64, inferer.input_shapes())
        };

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

fn test_fixed_batcher(
    onnx: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let instance = EpsilonInjector::wrap(
        fixed_batch_inferer_from_stream(&mut reader, sizes)?,
        "epsilon",
    )?;

    execute_steps(instance, "fixed", steps, batch_size)
}

fn test_dynamic_batcher(
    onnx: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let raw = batched_inferer_from_stream(&mut reader, sizes)?;
    let instance = EpsilonInjector::wrap(raw, "epsilon")?;

    execute_steps(instance, "dynamic", steps, batch_size)
}

fn test_no_batcher(onnx: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let raw = simple_inferer_from_stream(&mut reader)?;
    let instance = EpsilonInjector::wrap(raw, "epsilon")?;

    execute_steps(instance, "none", steps, batch_size)
}

pub(crate) fn execute_comparison(onnx_path: &Path, config: BatcherComparison) -> Result<()> {
    let fixed = test_fixed_batcher(
        onnx_path,
        &config.fixed_sizes,
        config.steps,
        config.batch_size,
    )?;
    let dynamic = test_dynamic_batcher(
        onnx_path,
        &config.dynamic_sizes,
        config.steps,
        config.batch_size,
    )?;
    let unbatched = test_no_batcher(onnx_path, config.steps, config.batch_size)?;

    let mut file = std::fs::File::create(config.output_file)?;

    for series in [fixed, dynamic, unbatched] {
        perchance::seed_global(0xff00ff00ff00ff00ff00ff00ff00ff00u128);
        for row in series {
            let denom = if config.batch_size > 0 {
                config.batch_size as f64
            } else {
                perchance::global().uniform_range_usize(1..10) as f64
            };

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
