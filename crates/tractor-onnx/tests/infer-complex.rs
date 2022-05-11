// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 10 May 2022

use tractor::{EpsilonInjector, Inferer};
use tractor_onnx::{
    batched_inferer_from_stream, fixed_batch_inferer_from_stream, simple_inferer_from_stream,
};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_infer_once_complex() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let mut instance =
        EpsilonInjector::wrap(simple_inferer_from_stream(&mut reader).unwrap(), "epsilon").unwrap();

    let observations = helpers::build_inputs_from_desc(1, instance.input_shapes());
    let result = instance.infer(observations);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].response["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_batched() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let mut instance = EpsilonInjector::wrap(
        batched_inferer_from_stream(&mut reader, &[10]).unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(10, instance.input_shapes());
    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].response["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_batched_not_loaded() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let mut instance = EpsilonInjector::wrap(
        batched_inferer_from_stream(&mut reader, &[5]).unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(10, instance.input_shapes());
    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].response["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_fixed_batch() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let mut instance = EpsilonInjector::wrap(
        fixed_batch_inferer_from_stream(&mut reader, &[4, 2, 1]).unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(7, instance.input_shapes());
    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 7);
    assert_eq!(result[&0].response["tanh_stretch"].len(), 36);
}
