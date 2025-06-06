// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_core::prelude::Inferer;

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_load_nnef_complex() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();
    cervo_nnef::builder(&mut reader)
        .build_basic()
        .expect("loading success");
}

#[test]
fn test_load_input_shape_complex() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();
    let instance = cervo_nnef::builder(&mut reader)
        .build_basic()
        .expect("failed reading instance");

    assert_eq!(
        instance.raw_input_shapes()[0].0,
        "features",
        "mismatch in input names"
    );

    assert_eq!(
        instance.raw_input_shapes()[0].1,
        vec![228],
        "mismatch in input shapes"
    );

    assert_eq!(
        instance.raw_input_shapes()[1].0,
        "images",
        "mismatch in input names"
    );

    assert_eq!(
        instance.raw_input_shapes()[1].1,
        vec![30, 30, 2],
        "mismatch in input shapes"
    );

    assert_eq!(
        instance.raw_input_shapes()[2].0,
        "epsilon",
        "mismatch in input names"
    );

    assert_eq!(
        instance.raw_input_shapes()[2].1,
        vec![36],
        "mismatch in input shapes"
    );
}

#[test]
fn test_load_output_shape_complex() {
    let mut reader = helpers::get_file("test-complex.nnef.tar").unwrap();
    let instance = cervo_nnef::builder(&mut reader).build_basic().unwrap();

    assert_eq!(
        instance.raw_output_shapes()[0].1,
        [36],
        "mismatch in output shape",
    );
}
