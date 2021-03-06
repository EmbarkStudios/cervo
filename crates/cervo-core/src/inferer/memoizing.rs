use super::{helpers, Inferer, Response, State};
use crate::model_api::ModelApi;
use anyhow::{Error, Result};
use std::collections::{hash_map::Entry, HashMap};
use tract_core::prelude::*;
use tract_hir::prelude::*;

/// The dynamic memoizing batch inferer generates execution plans to
/// fit each batch perfectly, achieving near-perfect performance no
/// matter how much data you have - with a hefty up-front cost for
/// each new batch size.
///
/// The dynamic batcher has the highest potential throughput when the
/// amount of data isn't known. By dynamically generating execution
/// plans to fit the exact amount of elements in each batch, it will
/// give tract optimal knowledge for execution each time. The downside
/// of this is that setting up a new plan is fairly costly, so doing
/// this for a batch size that is only seen once will waste memory and
/// compute resources.
///
/// While plans are cached; this still means that if your expected
/// batch size is can vary greatly, you'll end up with noticeable
/// spikes each time a new plan is generated. If you know you'll have
/// one or a few batch sizes - but not the exact size - this batcher
/// will end up providing good value and inform tuning for a fixed
/// batcher later.
///
/// If you know some batch sizes but not all, you can preload the
/// batcher with those plans to avoid having to build them at runtime.
///
/// # Pros
///
/// * Optimal amortized performance without tuning
/// * Requires no tuning for good results
///
/// # Cons
///
/// * For small amounts of data and large models the spikes can offset
/// amortized gains signifcantly

pub struct MemoizingDynamicInferer {
    symbol: Symbol,
    model: TypedModel,
    model_api: ModelApi,
    model_cache: HashMap<usize, TypedSimplePlan<TypedModel>>,
}

impl MemoizingDynamicInferer {
    /// Create an inferer for the provided `inference` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_model(model: InferenceModel, preloaded_sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelApi::for_model(&model)?;

        let (symbol, model) = helpers::build_symbolic_model(model, &model_api.inputs)?;
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

    /// Create an inferer for the provided `typed` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_typed(mut model: TypedModel, preloaded_sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelApi::for_typed_model(&model)?;

        let symbol = helpers::build_symbolic_typed(&mut model)?;
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

    fn infer_batched(&mut self, obs: Vec<State>, vec_out: &mut [Response]) -> TractResult<()> {
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
                vec_out[response_idx].data.insert(name.to_owned(), value);
            }
        }

        Ok(())
    }
}

impl Inferer for MemoizingDynamicInferer {
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
