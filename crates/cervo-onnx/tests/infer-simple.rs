// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

/*!

*/

use cervo_core::prelude::{EpsilonInjector, Inferer};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_infer_once_simple() {
    let mut reader = helpers::get_file("test.onnx").unwrap();
    let instance = cervo_onnx::builder(&mut reader).build_basic().unwrap();
    let mut instance = EpsilonInjector::wrap(instance, "epsilon").unwrap();

    let observations = helpers::build_inputs_from_desc(1, instance.input_shapes());

    let result = instance.infer(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].data["Identity"].len(), 1);
}
