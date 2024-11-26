use cervo_onnx::to_nnef;
use std::{thread, time::Duration};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_to_nnef_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();

    let result = to_nnef(&mut reader, None, false);
    result.unwrap();
}

#[test]
fn test_to_nnef_complex() {
    let mut reader = helpers::get_file("test-complex.onnx").unwrap();

    let result = to_nnef(&mut reader, None, false);
    result.unwrap();
}

#[test]
fn test_to_nnef_large() {
    let mut reader = helpers::get_file("test-large.onnx").unwrap();

    let result = to_nnef(&mut reader, None, false);
    result.unwrap();
}

#[test]
fn test_to_nnef_deterministic() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let result = to_nnef(&mut reader, None, true).unwrap();

    thread::sleep(Duration::from_secs(2));
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let result2 = to_nnef(&mut reader, None, true).unwrap();

    assert_eq!(result, result2);
}

#[test]
fn test_to_nnef_not_deterministic() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let result = to_nnef(&mut reader, None, false).unwrap();

    thread::sleep(Duration::from_secs(2));
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let result2 = to_nnef(&mut reader, None, false).unwrap();

    assert_ne!(result, result2);
}
