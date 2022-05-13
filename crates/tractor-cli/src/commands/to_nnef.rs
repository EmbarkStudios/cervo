// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 13 May 2022

/*!

*/
use anyhow::{bail, Result};
use clap::Parser;
use std::{fs::File, io::Write, path::PathBuf};

/// Convert ONNX files to NNEF.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct ToNnefArgs {
    in_files: Vec<PathBuf>,
}

pub(super) fn batch_onnx_to_nnef(config: ToNnefArgs) -> Result<()> {
    for file in &config.in_files {
        match file.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext == "onnx" => {}
            Some(ext) => bail!("unexpected extension: {:?}", ext),
            None => bail!("file without extension: {:?}", file),
        }
    }

    let mut tempfiles = vec![];
    for file in &config.in_files {
        let mut reader = File::open(file)?;
        let bytes = tractor_onnx::to_nnef(&mut reader)?;

        let mut out = tempfile::NamedTempFile::new()?;
        out.write_all(&bytes)?;

        let new_name = file.with_extension("nnef");
        tempfiles.push((new_name, out));
    }

    for (path, file) in tempfiles {
        std::fs::rename(&file, path)?;
    }

    Ok(())
}
