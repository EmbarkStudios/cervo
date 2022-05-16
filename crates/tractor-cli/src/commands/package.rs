// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

/*!

*/

use anyhow::{bail, Result};
use cervo_asset::{AssetData, AssetKind};
use clap::Parser;
use std::path::PathBuf;

/// Package the file into a crvo file.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct PackageArgs {
    /// The file to load. Must be .onnx or .nnef.tar
    infile: PathBuf,
    /// The file to write. Extensions are ignored.
    outfile: PathBuf,

    /// If set and reading an ONNX file, convert to NNEF on packaging.
    #[clap(short = 'O', long = "optimize")]
    optimize: bool,

    /// If provided, specialize the model for the batch size.
    #[clap(short = 'N', long = "batch-size", requires = "optimize")]
    batch_size: Option<usize>,
}

pub(super) fn package(config: PackageArgs) -> Result<()> {
    let kind = if tractor_nnef::is_nnef_tar(&config.infile) {
        AssetKind::Nnef
    } else {
        match config.infile.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => AssetKind::Onnx,
            Some(other) => bail!("unknown file type {:?}", other),
            None => bail!("missing file extension {:?}", config.infile),
        }
    };

    let bytes = std::fs::read(&config.infile)?;
    let asset = AssetData::new(kind, bytes);

    let asset = match asset.kind() {
        AssetKind::Onnx => {
            if config.optimize {
                asset.as_nnef(config.batch_size)?
            } else {
                asset
            }
        }
        AssetKind::Nnef => {
            if config.optimize {
                bail!("cannot optimize NNEF assets further")
            } else {
                asset
            }
        }
    };

    let file = config.outfile.with_extension("crvo");
    let data = asset.serialize()?;
    std::fs::write(&file, data)?;
    Ok(())
}
