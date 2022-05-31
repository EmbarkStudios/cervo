/*!
A basic unbatched inferer that doesn't require a lot of custom setup or management.
 */
use super::{Inferer, Response, State};
use crate::model_api::ModelApi;
use anyhow::{Error, Result};
use std::collections::HashMap;
use tract_core::{ndarray::IntoDimension, prelude::*};
use tract_hir::prelude::*;

use super::helpers;

/// The most basic inferer provided will deal with a single element at
/// a time, at the cost of reduced (but predictable) performance per
/// element.
///
/// # Pros
///
/// * Requires no tuning
/// * Very predictable performance across different workloads
///
/// # Cons
///
/// * Scales linearly unless it's the only code executing
pub struct BasicInferer {
    model: TypedSimplePlan<TypedModel>,
    model_api: ModelApi,
}

impl BasicInferer {
    /// Create an inferer for the provided `inference` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_model(model: InferenceModel) -> TractResult<Self> {
        let model_api = ModelApi::for_model(&model)?;
        let model = helpers::build_model(model, &model_api.inputs, 1i32)?;

        Ok(Self { model, model_api })
    }

    pub fn from_typed(model: TypedModel) -> TractResult<Self> {
        let model_api = ModelApi::for_typed_model(&model)?;
        let model = helpers::build_typed(model, 1i32)?;

        Ok(Self { model, model_api })
    }

    fn build_inputs(&mut self, mut obs: State) -> TVec<Tensor> {
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

    /// Run a single pass through the model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::plan::SimplePlan::run`] call.
    pub fn infer_once(&mut self, obs: State) -> TractResult<Response> {
        let inputs = self.build_inputs(obs);

        // Run the optimized plan to get actions back!
        let result = self.model.run(inputs)?;

        let mut response = Response::default();
        for (idx, (name, _)) in self.model_api.outputs.iter().enumerate() {
            response.data.insert(
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
}

impl Inferer for BasicInferer {
    fn infer(
        &mut self,
        observations: HashMap<u64, State>,
    ) -> Result<HashMap<u64, Response>, Error> {
        let mut responses = HashMap::default();
        for (id, obs) in observations.into_iter() {
            let response = self.infer_once(obs)?;
            responses.insert(id, response);
        }

        Ok(responses)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.inputs
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.outputs
    }
}
