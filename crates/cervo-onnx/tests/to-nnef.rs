// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_onnx::to_nnef;

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_to_nnef_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();

    let result = to_nnef(&mut reader, None);
    result.unwrap();
}

#[test]
fn test_to_nnef_complex() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let result = to_nnef(&mut reader, None);
    result.unwrap();
}

#[test]
fn test_to_nnef_large() {
    let mut reader = helpers::get_file("test-large.onnx").unwrap();

    let result = to_nnef(&mut reader, None);
    result.unwrap();
}
