// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_core::Inferer;
use cervo_onnx::simple_inferer_from_stream;

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_load_onnx_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    simple_inferer_from_stream(&mut reader).expect("loading success");
}

#[test]
fn test_load_input_shape_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let instance = simple_inferer_from_stream(&mut reader).expect("failed reading instance");
    assert_eq!(
        instance.input_shapes()[0].1,
        [2],
        "mismatch in input shapes"
    );
}

#[test]
fn test_load_output_shape_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let instance = simple_inferer_from_stream(&mut reader).unwrap();

    assert_eq!(
        instance.output_shapes()[0].1,
        &[1],
        "mismatch in output shapes",
    );
}
