// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2019, Embark Studios, all rights reserved.
// Created: 31 October 2019

#![warn(clippy::all)]
#![warn(rust_2018_idioms)]

use anyhow::{bail, Error};
use tractor::inferer::{Inferer, Observation};
use tractor::tract_wrap::TractInstance;
use tractor_onnx::{batched_inferer_from_stream, inferer_from_stream};

use std::ascii::AsciiExt;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

use structopt::StructOpt;

fn try_load_local_model(filename: &str, bs: usize) -> Result<TractInstance, Error> {
    let mut file = File::open(filename)?;

    let tract = if bs == 1 {
        inferer_from_stream(&mut file)
    } else {
        batched_inferer_from_stream(&mut file, &[1, bs])
    };

    if tract.is_ok() {
        Ok(tract.unwrap())
    } else {
        bail!(
            "Failed to load model from file: {}, {:?}",
            filename,
            tract.err()
        );
    }
}

#[derive(Debug, StructOpt)]
enum MeasureMode {
    BatchScaling,
    PerStep,
}

impl std::str::FromStr for MeasureMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "steptime" => Ok(MeasureMode::PerStep),
            "batching" => Ok(MeasureMode::BatchScaling),
            _ => Err(anyhow::anyhow!("unknown measure mode: {:?}", s)),
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "foo")]
struct RustyPerf {
    file: String,
    mode: MeasureMode,
}

fn measure_per_step_time(file: &str, count: u64, observations: &HashMap<u64, Observation>) {
    for bs in 1..count + 2 {
        let mut instance = try_load_local_model(file, bs as usize).unwrap();

        for step in 0..10 {
            let o = observations.clone();
            let start = Instant::now();
            instance.infer(o).unwrap();
            let elapsed = start.elapsed();

            println!(
                "{},{},{},{}",
                step,
                bs,
                count,
                elapsed.as_nanos() / (u128::from(count) * 1000),
            );
        }
    }
}

fn measure_time_per_element_batched(
    file: &str,
    count: u64,
    observations: &HashMap<u64, Observation>,
) {
    for bs in 1..count + 2 {
        let mut instance = try_load_local_model(file, bs as usize).unwrap();

        let start = Instant::now();

        for _ in 0..10 {
            instance.infer(observations.clone()).unwrap();
        }
        let elapsed = start.elapsed();

        eprintln!(
            "\t{},\t{},\t{}",
            bs,
            count,
            elapsed.as_nanos() / (10 * u128::from(count)) / 1000,
        );
    }
}

fn main() {
    let args = RustyPerf::from_args();

    let instance = try_load_local_model(&args.file, 1).unwrap();

    let inputs = instance.input_shapes();

    let data: HashMap<_, _> = inputs
        .iter()
        .map(|(key, count)| (key.clone(), vec![0.0; count.iter().fold(1, |a, b| a * b)]))
        .collect();

    for count in 1..11_u64 {
        let observations: HashMap<_, _> = (0..count)
            .map(|index| {
                let obs = Observation { data: data.clone() };
                (index, obs)
            })
            .collect();

        match args.mode {
            MeasureMode::BatchScaling => {
                measure_time_per_element_batched(&args.file, count, &observations)
            }
            MeasureMode::PerStep => measure_per_step_time(&args.file, count, &observations),
        }
    }
}
