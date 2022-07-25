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
use cervo_core::prelude::{Inferer, InfererExt};
use clap::Parser;

fn black_box<T>(dummy: T) -> T {
    unsafe { std::ptr::read_volatile(&dummy) }
}

#[derive(Parser, Debug)]
pub(crate) struct BatcherComparison {
    #[clap(long = "onnx", short = 'o')]
    onnx: Option<PathBuf>,
    #[clap(long = "nnef", short = 'n')]
    nnef: Option<PathBuf>,

    #[clap(short = 'f', long = "--fixed", use_value_delimiter = true)]
    fixed_sizes: Vec<usize>,
    #[clap(short = 'd', long = "--dynamic", use_value_delimiter = true)]
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
    let inputs = inferer.input_shapes().to_vec();
    let observations = crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs);

    let mut measurements = vec![];
    for step in 0..steps {
        let obs = if batch_size > 0 {
            observations.clone()
        } else {
            let batch_size = perchance::global().uniform_range_usize(1..10);
            crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs)
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

fn test_fixed_batcher_onnx(
    onnx: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let instance = cervo_onnx::builder(&mut reader)
        .build_fixed(sizes)?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "fixed+onnx", steps, batch_size)
}

fn test_dynamic_batcher_onnx(
    onnx: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let instance = cervo_onnx::builder(&mut reader)
        .build_memoizing(sizes)?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "dynamic+onnx", steps, batch_size)
}

fn test_no_batcher_onnx(onnx: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(onnx).unwrap();

    let instance = cervo_onnx::builder(&mut reader)
        .build_basic()?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "none+onnx", steps, batch_size)
}

fn test_fixed_batcher_nnef(
    nnef: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(nnef).unwrap();
    let instance = cervo_nnef::builder(&mut reader)
        .build_fixed(sizes)?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "fixed+nnef", steps, batch_size)
}

fn test_dynamic_batcher_nnef(
    nnef: &Path,
    sizes: &[usize],
    steps: usize,
    batch_size: usize,
) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(nnef).unwrap();

    let instance = cervo_nnef::builder(&mut reader)
        .build_memoizing(sizes)?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "dynamic+nnef", steps, batch_size)
}

fn test_no_batcher_nnef(nnef: &Path, steps: usize, batch_size: usize) -> Result<Vec<Measurement>> {
    let mut reader = crate::helpers::get_file(nnef).unwrap();

    let instance = cervo_nnef::builder(&mut reader)
        .build_basic()?
        .with_default_epsilon("epsilon")?;

    execute_steps(instance, "none+nnef", steps, batch_size)
}

pub(crate) fn execute_comparison(config: BatcherComparison) -> Result<()> {
    let mut file = std::fs::File::create(config.output_file)?;
    if let Some(onnx_path) = config.onnx {
        let fixed = test_fixed_batcher_onnx(
            &onnx_path,
            &config.fixed_sizes,
            config.steps,
            config.batch_size,
        )?;
        let dynamic = test_dynamic_batcher_onnx(
            &onnx_path,
            &config.dynamic_sizes,
            config.steps,
            config.batch_size,
        )?;
        let unbatched = test_no_batcher_onnx(&onnx_path, config.steps, config.batch_size)?;

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
    }

    if let Some(nnef_path) = config.nnef {
        let fixed = test_fixed_batcher_nnef(
            &nnef_path,
            &config.fixed_sizes,
            config.steps,
            config.batch_size,
        )?;
        let dynamic = test_dynamic_batcher_nnef(
            &nnef_path,
            &config.dynamic_sizes,
            config.steps,
            config.batch_size,
        )?;
        let unbatched = test_no_batcher_nnef(&nnef_path, config.steps, config.batch_size)?;

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
    }

    Ok(())
}
