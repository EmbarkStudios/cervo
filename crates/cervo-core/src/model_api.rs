// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

use tract_core::{model::TypedModel, tract_data::TractResult};
use tract_hir::{infer::Factoid, prelude::InferenceModel};

/// The `ModelApi` describes the inputs and outputs for a model.
#[derive(Debug)]
pub struct ModelApi {
    /// The named model inputs.
    pub inputs: Vec<(String, Vec<usize>)>,

    /// The named model outputs.
    pub outputs: Vec<(String, Vec<usize>)>,
}

impl ModelApi {
    /// Extract the model API from the provided inference model.
    pub fn for_model(model: &InferenceModel) -> TractResult<Self> {
        let mut inputs: Vec<(String, Vec<usize>)> = Default::default();
        for input_outlet in model.input_outlets()? {
            let node = model.node(input_outlet.node);
            let name = node.name.split(':').next().unwrap().to_owned();
            let input_shape = &model.input_fact(input_outlet.node)?.shape;

            inputs.push((
                name,
                input_shape
                    .dims()
                    .filter_map(|value| value.concretize().and_then(|v| v.to_i64().ok()))
                    .map(|val| val as usize)
                    .collect(),
            ));
        }

        let mut outputs: Vec<(String, Vec<usize>)> = Default::default();
        for (idx, output_outlet) in model.output_outlets().unwrap().iter().enumerate() {
            let name = model.outlet_labels[output_outlet]
                .split(':')
                .next()
                .unwrap()
                .to_owned();

            let output_shape = &model.output_fact(idx)?.shape;
            outputs.push((
                name,
                output_shape
                    .dims()
                    .filter_map(|value| value.concretize().and_then(|v| v.to_i64().ok()))
                    .map(|val| val as usize)
                    .collect(),
            ));
        }

        Ok(Self { outputs, inputs })
    }

    // Note[TS]: Clippy wants us to use name...clone_into(&name) but that's illegal.
    #[allow(clippy::assigning_clones)]
    pub fn for_typed_model(model: &TypedModel) -> TractResult<Self> {
        let mut inputs: Vec<(String, Vec<usize>)> = Default::default();

        for input_outlet in model.input_outlets()? {
            let node = model.node(input_outlet.node);
            let mut name = node.name.split(':').next().unwrap().to_owned();
            if name.ends_with("_0") {
                name = name.strip_suffix("_0").unwrap().to_owned();
            }
            let input_shape = &model.input_fact(input_outlet.node)?.shape;

            inputs.push((
                name,
                input_shape
                    .iter()
                    .filter_map(|dim| dim.to_i64().map(|v| v as usize).ok())
                    .collect(),
            ));
        }

        let mut outputs: Vec<(String, Vec<usize>)> = Default::default();

        for (idx, output_outlet) in model.outputs.iter().enumerate() {
            let mut name = model.outlet_labels[output_outlet]
                .split(':')
                .next()
                .unwrap()
                .to_owned();
            if name.ends_with("_0") {
                name = name.strip_suffix("_0").unwrap().to_owned();
            }

            let output_shape = &model.output_fact(idx)?.shape;
            let clean_shape = output_shape
                .iter()
                .filter_map(|dim| dim.to_i64().map(|v| v as usize).ok())
                .collect();

            outputs.push((name, clean_shape));
        }

        Ok(Self { outputs, inputs })
    }
}
