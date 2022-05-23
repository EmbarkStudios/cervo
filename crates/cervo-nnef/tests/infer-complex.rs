// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_core::{EpsilonInjector, Inferer};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_infer_once_complex() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();

    let mut instance = EpsilonInjector::wrap(
        cervo_nnef::builder(&mut reader).build_basic().unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(1, instance.input_shapes());
    let result = instance.infer(observations);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].data["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_batched() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();

    let mut instance = EpsilonInjector::wrap(
        cervo_nnef::builder(&mut reader)
            .build_memoizing(&[10])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(10, instance.input_shapes());
    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].data["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_batched_not_loaded() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();

    let mut instance = EpsilonInjector::wrap(
        cervo_nnef::builder(&mut reader)
            .build_memoizing(&[5])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(10, instance.input_shapes());
    let result = instance.infer(observations);
    eprintln!("result {:?}", result);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].data["tanh_stretch"].len(), 36);
}

#[test]
fn test_infer_once_complex_fixed_batch() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();

    let mut instance = EpsilonInjector::wrap(
        cervo_nnef::builder(&mut reader)
            .build_fixed(&[4, 2, 1])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let observations = helpers::build_inputs_from_desc(7, instance.input_shapes());
    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 7);
    assert_eq!(result[&0].data["tanh_stretch"].len(), 36);
}
