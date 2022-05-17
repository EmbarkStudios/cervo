// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!

*/

use anyhow::{bail, Result};
use cervo_asset::AssetData;
use cervo_core::Inferer;
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

    let model = if cervo_nnef::is_nnef_tar(&config.file) {
        cervo_nnef::simple_inferer_from_stream(&mut reader)?
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => cervo_onnx::simple_inferer_from_stream(&mut reader)?,
            Some("crvo") => AssetData::deserialize(&mut reader)?.load_simple()?,
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
