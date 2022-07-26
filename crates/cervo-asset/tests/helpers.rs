// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

#![allow(dead_code)]

use std::{collections::HashMap, fs::File, path};

use cervo_core::prelude::State;

pub fn get_file(name: &'static str) -> std::io::Result<File> {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut path = path::PathBuf::from(&crate_dir);
    path.push("..");
    path.push("..");
    path.push("brains");
    path.push(name);

    std::fs::File::open(path)
}

pub fn build_inputs_from_desc(count: u64, inputs: &[(String, Vec<usize>)]) -> HashMap<u64, State> {
    (0..count)
        .map(|idx| {
            (
                idx,
                State {
                    data: inputs
                        .iter()
                        .map(|(key, count)| (key.as_str(), vec![0.0; count.iter().product()]))
                        .collect(),
                },
            )
        })
        .collect()
}
