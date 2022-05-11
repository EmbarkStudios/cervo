// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2019, Embark Studios, all rights reserved.
// Created: 31 October 2019

#![warn(clippy::all)]
#![warn(rust_2018_idioms)]

mod compare_batchers;
mod helpers;

use anyhow::Error;
use compare_batchers::BatcherComparison;
use tractor::{Inferer, Observation};
use tractor_onnx::batched_inferer_from_stream;

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use structopt::StructOpt;

// fn try_load_local_model(filename: &str, bs: usize) -> Result<TractInstance, Error> {
//     let mut file = File::open(filename)?;

//     let tract = if bs == 1 {
//         inferer_from_stream(&mut file)
//     } else {
//         batched_inferer_from_stream(&mut file, &[1, bs])
//     };

//     tract.map_err(|err| anyhow::anyhow!("Failed to load model from file: {}, {:?}", filename, err))
// }

#[derive(Debug, StructOpt)]
enum MeasureMode {
    BatchScaling,
    PerStep,
    Batchers(BatcherComparison),
}

#[derive(Debug, StructOpt)]
#[structopt(name = "foo")]
struct RustyPerf {
    file: PathBuf,
    #[structopt(subcommand)] // Note that we mark a field as a subcommand
    mode: MeasureMode,
}

// fn measure_per_step_time(file: &str, count: u64, observations: &HashMap<u64, Observation>) {
//     for bs in 1..count + 2 {
//         let mut instance = try_load_local_model(file, bs as usize).unwrap();

//         for step in 0..10 {
//             let o = observations.clone();
//             let start = Instant::now();
//             instance.infer(o).unwrap();
//             let elapsed = start.elapsed();

//             println!(
//                 "{},{},{},{}",
//                 step,
//                 bs,
//                 count,
//                 elapsed.as_nanos() / (u128::from(count) * 1000),
//             );
//         }
//     }
// }

// fn measure_time_per_element_batched(
//     file: &str,
//     count: u64,
//     observations: &HashMap<u64, Observation>,
// ) {
//     for bs in 1..count + 2 {
//         let mut instance = try_load_local_model(file, bs as usize).unwrap();

//         let start = Instant::now();

//         for _ in 0..10 {
//             instance.infer(observations.clone()).unwrap();
//         }
//         let elapsed = start.elapsed();

//         eprintln!(
//             "\t{},\t{},\t{}",
//             bs,
//             count,
//             elapsed.as_nanos() / (10 * u128::from(count)) / 1000,
//         );
//     }
// }

fn main() {
    let args = RustyPerf::from_args();

    match args.mode {
        MeasureMode::BatchScaling => {
            //measure_time_per_element_batched(&args.file, count, &observations)
        }
        MeasureMode::PerStep => {} //measure_per_step_time(&args.file, count, &observations),
        MeasureMode::Batchers(config) => {
            compare_batchers::execute_comparison(&args.file, config).unwrap()
        }
    }
}
