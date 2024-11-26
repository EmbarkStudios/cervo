// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 13 May 2022

use anyhow::{bail, Result};
use clap::Parser;
use std::{fs::File, io::Write, path::PathBuf};

/// Convert ONNX files to NNEF.
///
/// Output files will be have same name as the ONNX file.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct BatchToNnefArgs {
    /// One or more input ONNX files.
    in_files: Vec<PathBuf>,

    /// The desired batch size. Default: a symbolic batch size.
    #[clap(short = 'b', long = "batch-size")]
    batch_size: Option<usize>,

    /// If set, will fix the timestamps in the nnef tar.
    #[clap(long = "deterministic")]
    deterministic: bool,
}

/// Convert an ONNX file to NNEF.
#[derive(Parser, Debug)]
#[clap()]
pub(crate) struct ToNnefArgs {
    /// The source ONNX file
    in_file: PathBuf,

    /// The destination NNEF tar file
    out_file: PathBuf,

    /// The desired batch size. Default: a symbolic batch size.
    #[clap(short = 'b', long = "batch-size")]
    batch_size: Option<usize>,

    /// If set, will fix the timestamps in the nnef tar.
    #[clap(long = "deterministic")]
    deterministic: bool,
}

pub(super) fn onnx_to_nnef(config: ToNnefArgs) -> Result<()> {
    let ToNnefArgs {
        in_file,
        out_file,
        batch_size,
        deterministic,
    } = config;

    match in_file.extension().and_then(|ext| ext.to_str()) {
        Some("onnx") => {}
        Some(ext) => bail!("unexpected extension: {:?}", ext),
        None => bail!("file without extension: {:?}", in_file),
    }

    match cervo::nnef::is_nnef_tar(&out_file) {
        true => {}
        false => bail!("unexpected extension: {:?}", out_file),
    }

    let mut reader = File::open(in_file)?;
    let mut bytes = cervo::onnx::to_nnef(&mut reader, batch_size, deterministic)?;
    bytes.shrink_to_fit();

    let mut out = tempfile::NamedTempFile::new()?;
    out.write_all(&bytes)?;

    std::fs::copy(&out, out_file)?;

    Ok(())
}

pub(super) fn batch_onnx_to_nnef(config: BatchToNnefArgs) -> Result<()> {
    for file in &config.in_files {
        match file.extension().and_then(|ext| ext.to_str()) {
            Some("onnx") => {}
            Some(ext) => bail!("unexpected extension: {:?}", ext),
            None => bail!("file without extension: {:?}", file),
        }
    }

    for in_file in config.in_files {
        let out_file = in_file.with_extension("nnef.tar");

        let args = ToNnefArgs {
            in_file,
            out_file,
            batch_size: config.batch_size,
            deterministic: config.deterministic,
        };

        onnx_to_nnef(args)?;
    }

    Ok(())
}
