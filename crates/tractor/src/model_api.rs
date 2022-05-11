// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 11 May 2022

/*!

*/

use tract_core::tract_data::TractResult;
use tract_hir::{infer::Factoid, prelude::InferenceModel};

pub struct ModelAPI {
    pub inputs: Vec<(String, Vec<usize>)>,
    pub outputs: Vec<(String, Vec<usize>)>,
}

impl ModelAPI {
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
                    .filter_map(|value| value.concretize().map(|v| v.to_i64().unwrap() as usize))
                    .collect(),
            ));
        }

        let mut outputs: Vec<(String, Vec<usize>)> = Default::default();

        for output_outlet in &model.outputs {
            let name = model.outlet_labels[output_outlet]
                .split(':')
                .next()
                .unwrap()
                .to_owned();

            let output_shape = &model.output_fact(output_outlet.slot)?.shape;

            outputs.push((
                name,
                output_shape
                    .dims()
                    .filter_map(|value| value.concretize().map(|v| v.to_i64().unwrap() as usize))
                    .collect(),
            ));
        }

        Ok(Self { outputs, inputs })
    }
}
