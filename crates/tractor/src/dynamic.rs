#![allow(clippy::explicit_counter_loop)]

use crate::model_api::ModelAPI;

use super::inferer::{Inferer, Observation, Response};
use anyhow::{bail, Error, Result};

use std::collections::{hash_map::Entry, HashMap};

use tract_core::prelude::*;
use tract_hir::prelude::*;

pub struct DynamicBatchingInferer {
    symbol: Symbol,
    model: TypedModel,

    model_api: ModelAPI,

    model_cache: HashMap<usize, TypedSimplePlan<TypedModel>>,
}

fn build_model(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
) -> Result<(Symbol, TypedModel)> {
    let s = Symbol::new('N');
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(s.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    // optimize the model and get an execution plan
    let model = model.into_typed()?.into_decluttered()?;
    Ok((s, model))
}

impl DynamicBatchingInferer {
    pub fn from_model(model: InferenceModel, preloaded_sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelAPI::for_model(&model)?;

        let (symbol, model) = build_model(model, &model_api.inputs)?;
        let mut this = Self {
            symbol,
            model,

            model_api,
            model_cache: Default::default(),
        };

        for size in preloaded_sizes {
            this.get_concrete_model(*size)?;
        }

        Ok(this)
    }

    fn build_inputs(&mut self, obs: Vec<Observation>) -> (TVec<Tensor>, usize) {
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

    fn get_concrete_model(&mut self, size: usize) -> Result<&TypedSimplePlan<TypedModel>> {
        if let Entry::Vacant(e) = self.model_cache.entry(size) {
            let p = self
                .model
                .concretize_dims(&SymbolValues::default().with(self.symbol, size as i64))?
                .into_optimized()?
                .into_decluttered()?
                .into_runnable()?;

            e.insert(p);
        }

        Ok(&self.model_cache[&size])
    }
    pub fn infer_batched(
        &mut self,
        obs: Vec<Observation>,
        vec_out: &mut Vec<Response>,
    ) -> TractResult<()> {
        let (inputs, count) = self.build_inputs(obs);

        // Run the optimized plan to get actions back!
        let result = self.get_concrete_model(count)?.run(inputs)?;

        for (idx, (name, shape)) in self.model_api.outputs.iter().enumerate() {
            for (response_idx, value) in result[idx]
                .to_array_view::<f32>()?
                .as_slice()
                .unwrap()
                .chunks(shape.iter().product())
                .map(|value| value.to_vec())
                .enumerate()
            {
                vec_out[response_idx]
                    .response
                    .insert(name.to_owned(), value);
            }
        }

        Ok(())
    }

    pub fn infer_tract(
        &mut self,
        observations: HashMap<u64, Observation>,
    ) -> TractResult<HashMap<u64, Response>> {
        let mut responses: Vec<Response> = vec![Response::default(); observations.len()];
        let (ids, obs): (Vec<_>, Vec<_>) = observations.into_iter().unzip();

        self.infer_batched(obs, &mut responses)?;

        let results: HashMap<u64, Response> = ids.into_iter().zip(responses.drain(..)).collect();

        Ok(results)
    }
}

impl Inferer for DynamicBatchingInferer {
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
