// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!

*/

use anyhow::{bail, Result};
use cervo::asset::AssetData;
use cervo::core::prelude::Inferer;
use clap::Parser;
use std::{fs::File, path::PathBuf};

/// Print API for a model
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct ApiArgs {
    file: PathBuf,
}

pub(super) fn describe_api(config: ApiArgs) -> Result<()> {
    let mut reader = File::open(&config.file)?;

    let model = if cervo::nnef::is_nnef_tar(&config.file) {
        cervo::nnef::builder(&mut reader).build_basic()?
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => cervo::onnx::builder(&mut reader).build_basic()?,
            Some("crvo") => AssetData::deserialize(&mut reader)?.load_basic()?,
            Some(other) => bail!("unknown file type {:?}", other),
            None => bail!("missing file extension {:?}", config.file),
        }
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
