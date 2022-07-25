// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

/*!

*/

use std::{collections::HashMap, fs::File, path::Path};

use cervo_core::prelude::State;

pub fn get_file<T: AsRef<Path>>(name: T) -> std::io::Result<File> {
    std::fs::File::open(name)
}

pub fn build_inputs_from_desc<'a>(
    count: u64,
    inputs: &'a [(String, Vec<usize>)],
) -> HashMap<u64, State<'a>> {
    (0..count)
        .map(|idx| {
            (
                idx,
                State {
                    data: inputs
                        .iter()
                        .filter(|(key, count)| key != "epsilon")
                        .map(|(key, count)| (key.as_str(), vec![0.0; count.iter().product()]))
                        .collect(),
                },
            )
        })
        .collect()
}
