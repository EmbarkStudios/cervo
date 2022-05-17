// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

/*!

*/

use std::{collections::HashMap, fs::File, path::Path};

use cervo_core::State;

pub fn get_file<T: AsRef<Path>>(name: T) -> std::io::Result<File> {
    std::fs::File::open(name)
}

pub fn build_inputs_from_desc(count: u64, inputs: &[(String, Vec<usize>)]) -> HashMap<u64, State> {
    (0..count)
        .map(|idx| {
            (
                idx,
                State {
                    data: inputs
                        .iter()
                        .map(|(key, count)| ((*key).clone(), vec![0.0; count.iter().product()]))
                        .collect(),
                },
            )
        })
        .collect()
}
