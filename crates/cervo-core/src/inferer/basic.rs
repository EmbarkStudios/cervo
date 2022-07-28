/*!
A basic unbatched inferer that doesn't require a lot of custom setup or management.
 */
use super::Inferer;
use crate::{batcher::ScratchPadView, model_api::ModelApi};
use anyhow::Result;
use tract_core::prelude::{tvec, TVec, Tensor, TractResult, TypedModel, TypedSimplePlan};
use tract_hir::prelude::InferenceModel;

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

    fn build_inputs(&mut self, obs: &ScratchPadView) -> Result<TVec<Tensor>> {
        let mut inputs = TVec::default();

        for (idx, (name, shape)) in self.model_api.inputs.iter().enumerate() {
            assert_eq!(name, obs.input_name(idx));

            let mut full_shape = tvec![1];
            full_shape.extend_from_slice(shape);

            let total_count: usize = full_shape.iter().product();
            assert_eq!(total_count, obs.input_slot(idx).len());

            let tensor = Tensor::from_shape(&full_shape, obs.input_slot(idx))?;

            inputs.push(tensor);
        }

        Ok(inputs)
    }
}

impl Inferer for BasicInferer {
    fn select_batch_size(&self, _: usize) -> usize {
        1
    }

    fn infer_raw(&mut self, mut pad: ScratchPadView) -> Result<(), anyhow::Error> {
        let inputs = self.build_inputs(&pad)?;

        // Run the optimized plan to get actions back!
        let result = self.model.run(inputs)?;

        for idx in 0..self.model_api.outputs.iter().len() {
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
