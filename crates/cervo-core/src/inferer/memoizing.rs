use super::{helpers, Inferer};
use crate::{batcher::ScratchPadView, model_api::ModelApi};
use anyhow::Result;
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    collections::{hash_map::Entry, HashMap},
    ops::Deref,
};
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
    model_cache: RwLock<HashMap<usize, TypedSimplePlan<TypedModel>>>,
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
        let this = Self {
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
        let this = Self {
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

    fn build_inputs(&self, batch: &ScratchPadView) -> Result<TVec<Tensor>> {
        let size = batch.len();

        let mut inputs = TVec::default();

        for (idx, (name, shape)) in self.model_api.inputs.iter().enumerate() {
            assert_eq!(name, batch.input_name(idx));

            let mut full_shape = tvec![size];
            full_shape.extend_from_slice(shape);

            let total_count: usize = full_shape.iter().product();
            assert_eq!(total_count, batch.input_slot(idx).len());

            let shape = full_shape;

            let tensor = Tensor::from_shape(&shape, batch.input_slot(idx))?;

            inputs.push(tensor);
        }

        Ok(inputs)
    }

    fn get_concrete_model(
        &self,
        size: usize,
    ) -> Result<impl Deref<Target = TypedSimplePlan<TypedModel>> + '_> {
        let cache = self.model_cache.upgradable_read();
        let cache = {
            if !cache.contains_key(&size) {
                let mut content = RwLockUpgradableReadGuard::upgrade(cache);
                if let Entry::Vacant(e) = content.entry(size) {
                    let p = self
                        .model
                        .concretize_dims(&SymbolValues::default().with(self.symbol, size as i64))?
                        .into_optimized()?
                        .into_decluttered()?
                        .into_runnable()?;

                    e.insert(p);
                }

                RwLockWriteGuard::downgrade(content)
            } else {
                RwLockUpgradableReadGuard::downgrade(cache)
            }
        };

        Ok(RwLockReadGuard::map(cache, |c| &c[&size]))
    }
}

impl Inferer for MemoizingDynamicInferer {
    fn select_batch_size(&self, max_count: usize) -> usize {
        max_count
    }

    fn infer_raw(&self, mut pad: ScratchPadView) -> Result<(), anyhow::Error> {
        let count = pad.len();
        let inputs = self.build_inputs(&pad)?;

        let result = self.get_concrete_model(count)?.run(inputs)?;

        for idx in 0..self.model_api.outputs.len() {
            let value = result[idx].as_slice::<f32>()?;
            pad.output_slot_mut(idx).copy_from_slice(value);
        }

        Ok(())
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.inputs
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.model_api.outputs
    }
}
