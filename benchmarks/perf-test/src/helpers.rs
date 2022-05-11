// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 11 May 2022

/*!

*/

use std::{
    collections::HashMap,
    fs::File,
    path::{self, Path},
};

use tractor::Observation;

pub fn get_file<T: AsRef<Path>>(name: T) -> std::io::Result<File> {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut path = path::PathBuf::from(&crate_dir);
    path.push(name);

    std::fs::File::open(path)
}

pub fn build_inputs_from_desc(
    count: u64,
    inputs: &[(String, Vec<usize>)],
) -> HashMap<u64, Observation> {
    (0..count)
        .map(|idx| {
            (
                idx,
                Observation {
                    data: inputs
                        .iter()
                        .map(|(key, count)| ((*key).clone(), vec![0.0; count.iter().product()]))
                        .collect(),
                },
            )
        })
        .collect()
}
