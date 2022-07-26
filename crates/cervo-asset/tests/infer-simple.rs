// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

/*!

*/

use cervo_asset::AssetData;
use cervo_core::prelude::{Inferer, InfererExt};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_infer_once_basic() {
    let mut reader = helpers::get_file("test.crvo").unwrap();
    let mut instance = AssetData::deserialize(&mut reader)
        .expect("a valid asset")
        .load_basic()
        .expect("success")
        .with_default_epsilon("epsilon")
        .unwrap();

    let shapes = instance.input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(1, &shapes);

    let result = instance.infer_batch(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].data["Identity"].len(), 1);
}

#[test]
fn test_infer_once_basic_nnef() {
    let mut reader = helpers::get_file("test-nnef.crvo").unwrap();
    let mut instance = AssetData::deserialize(&mut reader)
        .expect("a valid asset")
        .load_basic()
        .expect("an inferer")
        .with_default_epsilon("epsilon")
        .expect("a noise wrapper");

    let shapes = instance.input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(1, &shapes);
    let result = instance.infer_batch(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].data["Identity"].len(), 1);
}
