use cervo_asset::AssetData;
use std::{thread, time::Duration};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_to_nnef_raw() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    instance.to_nnef(None, false).unwrap();
}

#[test]
fn test_to_nnef_raw_nbs() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    instance.to_nnef(Some(10), false).unwrap();
}

#[test]
fn test_to_nnef() {
    let reader = helpers::get_file("test.crvo").unwrap();
    let instance = AssetData::deserialize(reader).expect("a valid asset");

    instance.to_nnef(None, false).unwrap();
}

#[test]
fn test_to_nnef_not_deterministic() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    let instance = instance.to_nnef(None, false).unwrap();

    thread::sleep(Duration::from_secs(4));

    let reader = helpers::get_file("test.onnx").unwrap();
    let instance2 =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");
    let instance2 = instance2.to_nnef(None, false).unwrap();

    assert_ne!(
        instance.serialize().unwrap(),
        instance2.serialize().unwrap(),
    );
}

#[test]
fn test_to_nnef_deterministic() {
    let reader = helpers::get_file("test.onnx").unwrap();
    let instance =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");

    let instance = instance.to_nnef(None, true).unwrap();

    thread::sleep(Duration::from_secs(4));

    let reader = helpers::get_file("test.onnx").unwrap();
    let instance2 =
        AssetData::from_reader(cervo_asset::AssetKind::Onnx, reader).expect("a valid asset");
    let instance2 = instance2.to_nnef(None, true).unwrap();

    assert_eq!(
        instance.serialize().unwrap(),
        instance2.serialize().unwrap()
    );
}

#[test]
fn test_to_nnef_fails() {
    let reader = helpers::get_file("test-nnef.crvo").unwrap();
    let instance = AssetData::deserialize(reader).expect("a valid asset");

    instance.to_nnef(None, false).unwrap_err();
}
