#![allow(clippy::explicit_counter_loop)]

use crate::model_api::ModelAPI;

use super::inferer::{Inferer, Observation, Response};
use anyhow::{bail, Error, Result};

use std::collections::HashMap;

use tract_core::{ndarray::IntoDimension, prelude::*};
use tract_hir::prelude::*;

fn build_model(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
) -> Result<TypedSimplePlan<TypedModel>> {
    let s = 1;
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(s.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    let model = model
        .into_optimized()?
        .into_decluttered()?
        .into_runnable()?;

    Ok(model)
}

pub struct BasicInferer {
    model: TypedSimplePlan<TypedModel>,

    model_api: ModelAPI,
}

impl BasicInferer {
    pub fn from_model(model: InferenceModel) -> TractResult<Self> {
        let model_api = ModelAPI::for_model(&model)?;
        let model = build_model(model, &model_api.inputs)?;

        Ok(Self { model, model_api })
    }

    fn build_inputs(&mut self, mut obs: Observation) -> TVec<Tensor> {
        let mut inputs = TVec::default();

        for (name, shape) in self.model_api.inputs.iter() {
            let mut full_shape = tvec![1];
            full_shape.extend_from_slice(shape);

            debug_assert!(obs.data.contains_key(name));

            let tensor = unsafe {
                tract_ndarray::Array::from_shape_vec_unchecked(
                    full_shape.into_dimension(),
                    obs.data.remove(name).unwrap(),
                )
                .into()
            };

            inputs.push(tensor);
        }

        inputs
    }

    pub fn infer_once(&mut self, obs: Observation) -> TractResult<Response> {
        let inputs = self.build_inputs(obs);

        // Run the optimized plan to get actions back!
        let result = self.model.run(inputs)?;

        let mut response = Response::default();
        for (idx, (name, _)) in self.model_api.outputs.iter().enumerate() {
            response.response.insert(
                name.clone(),
                result[idx]
                    .to_array_view::<f32>()?
                    .to_slice()
                    .unwrap()
                    .to_vec(),
            );
        }

        Ok(response)
    }

    pub fn infer_tract(
        &mut self,
        observations: HashMap<u64, Observation>,
    ) -> TractResult<HashMap<u64, Response>> {
        let mut responses = HashMap::default();
        for (id, obs) in observations.into_iter() {
            let response = self.infer_once(obs)?;
            responses.insert(id, response);
        }

        Ok(responses)
    }
}

impl Inferer for BasicInferer {
    fn infer(
        &mut self,
        observations: HashMap<u64, Observation>,
    ) -> Result<HashMap<u64, Response>, Error> {
        let tract_result = self.infer_tract(observations);
        match tract_result {
            Ok(results) => Ok(results),
            Err(error) => bail!(format!("{:?}", error)),
        }
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.inputs
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.outputs
    }
}
