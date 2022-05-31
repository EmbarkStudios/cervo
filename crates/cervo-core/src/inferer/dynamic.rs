use super::{helpers, Inferer, Response, State};
use crate::model_api::ModelApi;
use anyhow::{Error, Result};
use std::collections::HashMap;
use tract_core::prelude::*;
use tract_hir::prelude::*;

/// The dynamic inferer hits a spot between the raw simplicity of a [`crate::prelude::BasicInferer`] and the spikiness
/// of a [`crate::prelude::MemoizingDynamicInferer`]. Instead of explicitly concretizing models and caching them, it
/// relies on tracts internal concretization which leads to worse performance overall; but beating out the
/// [`crate::prelude::BasicInferer`].
///
/// # Pros
///
/// * Requires no tuning for OK results
/// * Fixed memory and fairly linear performance scaling
///
/// # Cons
///
/// * Small extra overhead for small extra performance
/// * Worst option for small batch sizes
pub struct DynamicInferer {
    model: TypedSimplePlan<TypedModel>,
    model_api: ModelApi,
}

impl DynamicInferer {
    /// Create an inferer for the provided `inference` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_model(model: InferenceModel) -> TractResult<Self> {
        let model_api = ModelApi::for_model(&model)?;

        let (_, model) = helpers::build_symbolic_model(model, &model_api.inputs)?;
        let this = Self {
            model: model.into_optimized()?.into_runnable()?,
            model_api,
        };

        Ok(this)
    }

    /// Create an inferer for the provided `typed` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_typed(mut model: TypedModel) -> TractResult<Self> {
        let model_api = ModelApi::for_typed_model(&model)?;

        let _ = helpers::build_symbolic_typed(&mut model)?;
        let this = Self {
            model: model.into_optimized()?.into_runnable()?,
            model_api,
        };

        Ok(this)
    }

    fn build_inputs(&mut self, obs: Vec<State>) -> (TVec<Tensor>, usize) {
        let size = obs.len();
        let mut inputs = TVec::default();
        let mut named_inputs = TVec::default();

        for (name, shape) in &self.model_api.inputs {
            let mut full_shape = tvec![size];
            full_shape.extend_from_slice(shape);

            let total_count = full_shape.iter().product();
            named_inputs.push((name, (full_shape, Vec::with_capacity(total_count))));
        }

        for observation in obs {
            for (name, (_, store)) in named_inputs.iter_mut() {
                store.extend_from_slice(&observation.data[*name]);
            }
        }

        for (_, (shape, store)) in named_inputs {
            let tensor = unsafe {
                tract_ndarray::Array::from_shape_vec_unchecked(shape.into_vec(), store).into()
            };

            inputs.push(tensor);
        }

        (inputs, size)
    }

    fn infer_batched(&mut self, obs: Vec<State>, vec_out: &mut [Response]) -> TractResult<()> {
        let (inputs, _count) = self.build_inputs(obs);

        // Run the optimized plan to get actions back!
        let result = self.model.run(inputs)?;

        for (idx, (name, shape)) in self.model_api.outputs.iter().enumerate() {
            for (response_idx, value) in result[idx]
                .to_array_view::<f32>()?
                .as_slice()
                .unwrap()
                .chunks(shape.iter().product())
                .map(|value| value.to_vec())
                .enumerate()
            {
                vec_out[response_idx].data.insert(name.to_owned(), value);
            }
        }

        Ok(())
    }
}

impl Inferer for DynamicInferer {
    fn infer(
        &mut self,
        observations: HashMap<u64, State>,
    ) -> Result<HashMap<u64, Response>, Error> {
        let mut responses: Vec<Response> = vec![Response::default(); observations.len()];
        let (ids, obs): (Vec<_>, Vec<_>) = observations.into_iter().unzip();

        self.infer_batched(obs, &mut responses)?;

        let results: HashMap<u64, Response> = ids.into_iter().zip(responses.drain(..)).collect();

        Ok(results)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.inputs
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.outputs
    }
}
