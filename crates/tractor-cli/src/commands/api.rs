// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 13 May 2022

/*!

*/

use anyhow::{bail, Result};
use clap::Parser;
use std::{fs::File, path::PathBuf};
use tractor::Inferer;

/// Print API for a model
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct ApiArgs {
    file: PathBuf,
}

pub(super) fn describe_api(config: ApiArgs) -> Result<()> {
    let mut reader = File::open(&config.file)?;

    let model = match config.file.extension().and_then(|ext| ext.to_str()) {
        Some("onnx") => tractor_onnx::simple_inferer_from_stream(&mut reader)?,
        Some("nnef") => tractor_nnef::simple_inferer_from_stream(&mut reader)?,
        other => bail!("unknown file type {:?}", other),
    };

    println!("Inputs:");
    for (name, shape) in model.input_shapes() {
        println!("\t{:40}: {:?}", name, shape);
    }

    println!("\nOutputs:");
    for (name, shape) in model.output_shapes() {
        println!("\t{:40}: {:?}", name, shape);
    }
    Ok(())
}
