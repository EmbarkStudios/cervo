// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo::Inferer;
use cervo_asset::AssetData;

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_load_input_shape_simple_onnx() {
    let mut reader = helpers::get_file("test.crvo").unwrap();
    let instance = AssetData::deserialize(&mut reader)
        .expect("failed reading instance")
        .load_simple()
        .expect("an inferer");

    assert_eq!(
        instance.input_shapes()[0].1,
        [2],
        "mismatch in input shapes"
    );
}

#[test]
fn test_load_input_shape_simple_nnef() {
    let mut reader = helpers::get_file("test-nnef.crvo").unwrap();
    let instance = AssetData::deserialize(&mut reader)
        .expect("failed reading instance")
        .load_simple()
        .expect("an inferer");

    assert_eq!(
        instance.input_shapes()[0].1,
        [2],
        "mismatch in input shapes"
    );
}
