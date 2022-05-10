// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 10 May 2022

/*!

*/

use std::path;
use tractor::inferer::{Inferer, Observation};
use tractor_onnx::inferer_from_stream;

#[test]
fn test_infer_once_simple() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = path::PathBuf::from(&crate_dir);
    path.push("test.onnx");
    let mut reader = std::fs::File::open(path).unwrap();
    let mut instance = inferer_from_stream(&mut reader).unwrap();

    let input = vec![Observation {
        data: [("vector_observation".to_owned(), vec![0.0, 0.0])]
            .iter()
            .cloned()
            .collect(),
    }];

    let observations = input
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();

    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].response["Identity"].len(), 1);
}
