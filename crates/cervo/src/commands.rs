// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!


*/

use anyhow::Result;
use clap::Parser;

mod api;
mod describe;
mod package;
mod to_nnef;

/// The command to run.
#[derive(Parser, Debug)]
pub(crate) enum Command {
    ToNnef(to_nnef::ToNnefArgs),
    BatchToNnef(to_nnef::BatchToNnefArgs),
    Api(api::ApiArgs),
    Package(package::PackageArgs),
    Describe(describe::DescribeArgs),
}

pub(crate) fn run(command: Command) -> Result<()> {
    match command {
        Command::ToNnef(config) => to_nnef::onnx_to_nnef(config),
        Command::BatchToNnef(config) => to_nnef::batch_onnx_to_nnef(config),
        Command::Api(config) => api::describe_api(config),
        Command::Describe(config) => describe::describe(config),
        Command::Package(config) => package::package(config),
    }
}
