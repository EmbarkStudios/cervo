// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!

*/

use anyhow::{bail, Result};
use cervo_asset::AssetData;
use clap::Parser;
use std::{fs::File, path::PathBuf};

/// Shortly describe the model file.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct DescribeArgs {
    file: PathBuf,
}

pub(super) fn describe(config: DescribeArgs) -> Result<()> {
    let mut reader = File::open(&config.file)?;

    if cervo_nnef::is_nnef_tar(&config.file) {
        println!("a NNEF file");
    } else {
        match config.file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => println!("an ONNX file"),
            Some("crvo") => {
                let asset = AssetData::deserialize(&mut reader)?;
                println!("a native cervo file containing {} data", asset.kind());
            }
            Some(other) => bail!("unknown file type {:?}", other),
            None => bail!("missing file extension {:?}", config.file),
        }
    }

    Ok(())
}
