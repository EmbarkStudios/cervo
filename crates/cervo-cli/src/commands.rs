// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!


*/

use anyhow::Result;
use clap::Parser;

mod api;
mod benchmark;
mod describe;
mod package;
mod run;
#[cfg(feature = "serve")]
mod serve;
mod to_nnef;

/// The command to run.
#[derive(Parser, Debug)]
pub(crate) enum Command {
    ToNnef(to_nnef::ToNnefArgs),
    BatchToNnef(to_nnef::BatchToNnefArgs),
    Api(api::ApiArgs),
    Package(package::PackageArgs),
    Describe(describe::DescribeArgs),
    Benchmark(benchmark::Args),
    Run(run::Args),
    #[cfg(feature = "serve")]
    Serve(serve::Args),
}

pub(crate) fn run(command: Command) -> Result<()> {
    match command {
        Command::ToNnef(config) => to_nnef::onnx_to_nnef(config),
        Command::BatchToNnef(config) => to_nnef::batch_onnx_to_nnef(config),
        Command::Api(config) => api::describe_api(config),
        Command::Describe(config) => describe::describe(config),
        Command::Package(config) => package::package(config),
        Command::Benchmark(config) => benchmark::run(config),
        Command::Run(config) => run::run(config),
        #[cfg(feature = "serve")]
        Command::Serve(config) => serve::serve(config),
    }
}
