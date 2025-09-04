// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 10 May 2022

use cervo_core::{
    prelude::{EpsilonInjector, Inferer, InfererExt},
    recurrent::{RecurrentInfo, RecurrentTracker},
};

#[path = "./helpers.rs"]
mod helpers;

#[test]
fn test_infer_once_recurrent() {
    let mut reader = helpers::get_file("test-recurrent.onnx").unwrap();

    let instance = EpsilonInjector::wrap(
        cervo_onnx::builder(&mut reader).build_basic().unwrap(),
        "epsilon",
    )
    .unwrap();

    let instance = RecurrentTracker::new(
        instance,
        vec![
            RecurrentInfo {
                inkey: "lstm_hidden_state.1".to_owned(),
                outkey: "lstm_hidden_state".to_owned(),
            },
            RecurrentInfo {
                inkey: "lstm_cell_state.1".to_owned(),
                outkey: "lstm_cell_state".to_owned(),
            },
        ],
    )
    .unwrap();

    instance.begin_agent(0);
    let shapes = instance.raw_input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(1, &shapes);
    let result = instance.infer_batch(observations);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[&0].data["actions"].len(), 22);
    assert_eq!(result[&0].data.len(), 1);

    instance.end_agent(0);
}

#[test]
fn test_infer_once_recurrent_batched() {
    let mut reader = helpers::get_file("test-recurrent.onnx").unwrap();

    let instance = EpsilonInjector::wrap(
        cervo_onnx::builder(&mut reader)
            .build_memoizing(&[10])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let instance = RecurrentTracker::new(
        instance,
        vec![
            RecurrentInfo {
                inkey: "lstm_hidden_state.1".to_owned(),
                outkey: "lstm_hidden_state".to_owned(),
            },
            RecurrentInfo {
                inkey: "lstm_cell_state.1".to_owned(),
                outkey: "lstm_cell_state".to_owned(),
            },
        ],
    )
    .unwrap();

    for idx in 0..10 {
        instance.begin_agent(idx);
    }
    let shapes = instance.raw_input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(10, &shapes);
    let result = instance.infer_batch(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].data["actions"].len(), 22);
}

#[test]
fn test_infer_once_recurrent_batched_not_loaded() {
    let mut reader = helpers::get_file("test-recurrent.onnx").unwrap();

    let instance = EpsilonInjector::wrap(
        cervo_onnx::builder(&mut reader)
            .build_memoizing(&[5])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let instance = RecurrentTracker::new(
        instance,
        vec![
            RecurrentInfo {
                inkey: "lstm_hidden_state.1".to_owned(),
                outkey: "lstm_hidden_state".to_owned(),
            },
            RecurrentInfo {
                inkey: "lstm_cell_state.1".to_owned(),
                outkey: "lstm_cell_state".to_owned(),
            },
        ],
    )
    .unwrap();
    for idx in 0..10 {
        instance.begin_agent(idx);
    }
    let shapes = instance.raw_input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(10, &shapes);
    let result = instance.infer_batch(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 10);
    assert_eq!(result[&0].data["actions"].len(), 22);
}

#[test]
fn test_infer_once_recurrent_fixed_batch() {
    let mut reader = helpers::get_file("test-recurrent.onnx").unwrap();

    let instance = EpsilonInjector::wrap(
        cervo_onnx::builder(&mut reader)
            .build_fixed(&[4, 2, 1])
            .unwrap(),
        "epsilon",
    )
    .unwrap();

    let instance = RecurrentTracker::new(
        instance,
        vec![
            RecurrentInfo {
                inkey: "lstm_hidden_state.1".to_owned(),
                outkey: "lstm_hidden_state".to_owned(),
            },
            RecurrentInfo {
                inkey: "lstm_cell_state.1".to_owned(),
                outkey: "lstm_cell_state".to_owned(),
            },
        ],
    )
    .unwrap();
    for idx in 0..7 {
        instance.begin_agent(idx);
    }
    let shapes = instance.raw_input_shapes().to_vec();
    let observations = helpers::build_inputs_from_desc(7, &shapes);
    let result = instance.infer_batch(observations);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.len(), 7);
    assert_eq!(result[&0].data["actions"].len(), 22);
}
