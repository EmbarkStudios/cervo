// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 13 May 2022

/*!


*/

use anyhow::Result;
use clap::Parser;

mod api;
mod to_nnef;

#[derive(Parser, Debug)]
#[clap()]
pub(crate) enum Command {
    ToNnef(to_nnef::ToNnefArgs),
    Api(api::ApiArgs),
}

pub(crate) fn run(command: Command) -> Result<()> {
    match command {
        Command::ToNnef(config) => to_nnef::batch_onnx_to_nnef(config),
        Command::Api(config) => api::describe_api(config),
    }
}
