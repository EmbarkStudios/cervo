use super::{helpers, Inferer};
use crate::{batcher::ScratchPadView, model_api::ModelApi};
use anyhow::Result;
use tract_core::{
    prelude::{tvec, TVec, Tensor, TractResult, TypedModel, TypedSimplePlan},
    value::TValue,
};
use tract_hir::prelude::InferenceModel;

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

    fn build_inputs(&self, batch: &ScratchPadView<'_>) -> Result<TVec<TValue>> {
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

            inputs.push(tensor.into());
        }

        Ok(inputs)
    }
}

impl Inferer for DynamicInferer {
    fn select_batch_size(&self, max_count: usize) -> usize {
        max_count
    }

    fn infer_raw(&self, mut pad: ScratchPadView<'_>) -> Result<(), anyhow::Error> {
        let inputs = self.build_inputs(&pad)?;

        // Run the optimized plan to get actions back!
        let result = self.model.run(inputs)?;

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
