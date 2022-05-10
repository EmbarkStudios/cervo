// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 10 May 2022

/*!

*/

use std::path;
use tractor_onnx::to_nnef;

#[test]
fn test_to_nnef_complex() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut path = path::PathBuf::from(&crate_dir);
    path.push("foo.onnx");
    let mut reader = std::fs::File::open(path).unwrap();

    let result = to_nnef(&mut reader);
    result.unwrap();
}
