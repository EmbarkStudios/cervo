#![allow(clippy::explicit_counter_loop)]

use super::inferer::{Inferer, Response, State};
use crate::model_api::ModelAPI;
use anyhow::{bail, Error, Result};
use std::collections::HashMap;
use tract_core::prelude::*;
use tract_hir::prelude::*;

pub struct FixedBatchingInferer {
    model_api: ModelAPI,
    models: Vec<BatchedModel>,
}

pub struct BatchedModel {
    size: usize,
    plan: TypedSimplePlan<TypedModel>,
}

impl BatchedModel {
    fn build_inputs<It: std::iter::Iterator<Item = State>>(
        &mut self,
        obs: &mut It,
        model_api: &ModelAPI,
    ) -> TVec<Tensor> {
        let size = self.size;
        let mut inputs = TVec::default();
        let mut named_inputs = TVec::default();

        for (name, shape) in &model_api.inputs {
            let mut full_shape = tvec![size];
            full_shape.extend_from_slice(shape);

            let total_count = full_shape.iter().product();
            named_inputs.push((name, (full_shape, Vec::with_capacity(total_count))));
        }

        for observation in obs.take(size) {
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

        inputs
    }

    fn execute<It: std::iter::Iterator<Item = State>>(
        &mut self,
        observations: &mut It,
        model_api: &ModelAPI,
        vec_out: &mut [Response],
    ) -> Result<()> {
        let inputs = self.build_inputs(observations, model_api);

        let result = self.plan.run(inputs)?;

        for (idx, (name, shape)) in model_api.outputs.iter().enumerate() {
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
fn build_model(
    mut model: InferenceModel,
    s: i32,
    inputs: &[(String, Vec<usize>)],
) -> Result<TypedSimplePlan<TypedModel>> {
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

impl FixedBatchingInferer {
    pub fn from_model(model: InferenceModel, sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelAPI::for_model(&model)?;

        let mut sizes = sizes.to_vec();
        if !sizes.contains(&1) {
            sizes.push(1);
        }
        sizes.sort_unstable();
        sizes.reverse();

        let models = sizes
            .into_iter()
            .map(|size| {
                build_model(model.clone(), size as i32, &model_api.inputs)
                    .map(|m| BatchedModel { size, plan: m })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { models, model_api })
    }

    pub fn infer_batched(&mut self, obs: Vec<State>, vec_out: &mut [Response]) -> TractResult<()> {
        let mut offset = 0;
        let mut count = obs.len();
        let mut obs = obs.into_iter();

        for plan in &mut self.models {
            while (count / plan.size) > 0 {
                plan.execute(&mut obs, &self.model_api, &mut vec_out[offset..])?;
                count -= plan.size;
                offset += plan.size;
            }
        }

        Ok(())
    }

    pub fn infer_tract(
        &mut self,
        observations: HashMap<u64, State>,
    ) -> TractResult<HashMap<u64, Response>> {
        let mut responses: Vec<Response> = vec![Response::default(); observations.len()];
        let (ids, obs): (Vec<_>, Vec<_>) = observations.into_iter().unzip();

        self.infer_batched(obs, &mut responses)?;

        let results: HashMap<u64, Response> = ids.into_iter().zip(responses.drain(..)).collect();

        Ok(results)
    }
}

impl Inferer for FixedBatchingInferer {
    fn infer(
        &mut self,
        observations: HashMap<u64, State>,
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
