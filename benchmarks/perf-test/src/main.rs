// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2019, Embark Studios, all rights reserved.
// Created: 31 October 2019

#![warn(clippy::all)]
#![warn(rust_2018_idioms)]

use clap::Parser;
mod compare_batchers;
mod compare_batchsize;
mod compare_loading;
mod compare_noise;
mod helpers;

use compare_batchers::BatcherComparison;
use compare_batchsize::BatchScaling;
use compare_loading::LoadComparison;
use compare_noise::NoiseComparison;

#[derive(Debug, Parser)]
enum MeasureMode {
    BatchScaling(BatchScaling),
    Batchers(BatcherComparison),
    Noise(NoiseComparison),
    Loading(LoadComparison),
}

#[derive(Debug, Parser)]
#[clap(name = "cervo perf-tests")]
struct RustyPerf {
    #[clap(subcommand)] // Note that we mark a field as a subcommand
    mode: MeasureMode,
}

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
    let args = RustyPerf::parse();

    match args.mode {
        MeasureMode::BatchScaling(config) => {
            compare_batchsize::compare_batch_scaling(config).unwrap()
        }
        MeasureMode::Batchers(config) => compare_batchers::execute_comparison(config).unwrap(),
        MeasureMode::Noise(config) => compare_noise::execute_comparison(config).unwrap(),
        MeasureMode::Loading(config) => {
            compare_loading::compare_loadtimes(config).unwrap();
        }
    }
}
