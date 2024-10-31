// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 16 May 2022

use std::io::Read;

use cervo_asset::{AssetData, AssetKind};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_load_onnx_asset() {
    let mut reader = helpers::get_file("test.crvo").unwrap();

    let mut raw_onnx = vec![];
    helpers::get_file("test.onnx")
        .unwrap()
        .read_to_end(&mut raw_onnx)
        .expect("a valid read");
    let instance = AssetData::deserialize(&mut reader).expect("valid asset");

    assert_eq!(instance.kind(), AssetKind::Onnx,);
    assert_eq!(instance.data(), raw_onnx,);
}

#[test]
fn test_load_nnef_asset() {
    let mut reader = helpers::get_file("test-nnef.crvo").unwrap();

    let mut raw_onnx = vec![];
    helpers::get_file("test.nnef.tar")
        .unwrap()
        .read_to_end(&mut raw_onnx)
        .expect("a valid read");
    let instance = AssetData::deserialize(&mut reader).expect("valid asset");

    assert_eq!(instance.kind(), AssetKind::Nnef,);
    assert_eq!(instance.data(), raw_onnx,);
}
