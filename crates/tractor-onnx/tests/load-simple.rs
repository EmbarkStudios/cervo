// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 10 May 2022

/*!

*/

use std::path;
use tractor::inferer::Inferer;
use tractor_onnx::inferer_from_stream;

#[test]
fn test_load_onnx_simple() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = path::PathBuf::from(&crate_dir);
    path.push("test.onnx");
    let mut reader = std::fs::File::open(path).unwrap();
    inferer_from_stream(&mut reader).expect("loading success");
}

#[test]
fn test_load_input_shape_simple() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = path::PathBuf::from(&crate_dir);
    path.push("test.onnx");
    let mut reader = std::fs::File::open(path).expect("failed reading file");
    let instance = inferer_from_stream(&mut reader).expect("failed reading instance");
    assert_eq!(
        instance.input_shapes()[0].1,
        [2],
        "mismatch in input shapes"
    );
}

#[test]
fn test_load_output_shape_simple() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = path::PathBuf::from(&crate_dir);
    path.push("test.onnx");
    let mut reader = std::fs::File::open(path).unwrap();
    let instance = inferer_from_stream(&mut reader).unwrap();

    assert_eq!(
        instance.output_shapes()[0].1,
        &[1],
        "mismatch in output shapes",
    );
}
