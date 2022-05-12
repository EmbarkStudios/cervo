/*!
A reliable batched inferer that is a good fit if you know how much data you'll have and want stable performance.

As an added bonus, it'll subdivide your data into minibatches if the batching doesn't fit perfectly. To make this ork,
it'll add a single-element mode as well to ensure all data is consumed - such as if you feed it 9 elements with a
configured batch size of 8.

You can configure a wide number of different batch sizes, and the largest one will be used. Note that the overhead for
execution still is fairly large, but this helps amortize some of that cost away. For example; if you use a setup of [1,
2, 4, 8] as your supported batch sizes a batch of 15 elements would run each plan once.
 */
use super::{Inferer, Response, State};
use crate::model_api::ModelAPI;
use anyhow::{Error, Result};
use std::collections::HashMap;
use tract_core::prelude::*;
use tract_hir::prelude::*;

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

/// The fixed batch inferer provided will subdivide your data into minibatches to efficiently use a set of preconfigured minibatch-sizes.
///
/// # Pros
///
/// * Good and predictable performance if you know amount of data
/// * Flexible if you sometimes get extra data to deal with
///
/// # Cons
///
/// * Mini-batches lead to noticeable performance degradation
pub struct FixedBatchingInferer {
    model_api: ModelAPI,
    models: Vec<BatchedModel>,
}

impl FixedBatchingInferer {
    /// Create an inferer for the provided `inference` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
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

    fn infer_batched(&mut self, obs: Vec<State>, vec_out: &mut [Response]) -> TractResult<()> {
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
}

impl Inferer for FixedBatchingInferer {
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

struct BatchedModel {
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
