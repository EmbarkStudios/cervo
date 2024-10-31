// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_asset::AssetData;

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_to_nnef_raw() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    instance.to_nnef(None).unwrap();
}

#[test]
fn test_to_nnef_raw_nbs() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    instance.to_nnef(Some(10)).unwrap();
}

#[test]
fn test_to_nnef() {
    let reader = helpers::get_file("test.crvo").unwrap();
    let instance = AssetData::deserialize(reader).expect("a valid asset");

    instance.to_nnef(None).unwrap();
}

#[test]
fn test_to_nnef_fails() {
    let reader = helpers::get_file("test-nnef.crvo").unwrap();
    let instance = AssetData::deserialize(reader).expect("a valid asset");

    instance.to_nnef(None).unwrap_err();
}
