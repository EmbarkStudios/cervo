use super::{helpers, Batch, BatchResponse, Inferer};
use crate::{batcher::ScratchPadView, model_api::ModelApi};
use anyhow::{Context, Result};

use tract_core::prelude::*;
use tract_hir::prelude::*;

/// A reliable batched inferer that is a good fit if you know how much data you'll have and want stable performance.
///
/// As an added bonus, it'll subdivide your data into minibatches if the batching doesn't fit perfectly. To make this
/// work, it'll add a single-element mode as well to ensure all data is consumed - such as if you feed it 9 elements
/// with a configured batch size of 8.
///
/// You can configure a wide number of different batch sizes, and the largest one will be used. Note that the overhead for
/// execution still is fairly large, but this helps amortize some of that cost away. For example; if you use a setup of [1,
/// 2, 4, 8] as your supported batch sizes a batch of 15 elements would run each plan once.
///
/// # Pros
///
/// * Good and predictable performance if you know amount of data
/// * Flexible if you sometimes get extra data to deal with
///
/// # Cons
///
/// * Mini-batches add overhead
/// * Diminishing returns on each supported batch size.
pub struct FixedBatchInferer {
    model_api: ModelApi,
    models: Vec<BatchedModel>,
}

fn fixup_sizes(sizes: &[usize]) -> Vec<usize> {
    let mut sizes = sizes.to_vec();
    if !sizes.contains(&1) {
        sizes.push(1);
    }
    sizes.sort_unstable();
    sizes.reverse();

    sizes
}

impl FixedBatchInferer {
    /// Create an inferer for the provided `inference` model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_model(model: InferenceModel, sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelApi::for_model(&model)?;

        let sizes = fixup_sizes(sizes);

        let models = sizes
            .into_iter()
            .map(|size| {
                helpers::build_model(model.clone(), &model_api.inputs, size as i32)
                    .map(|m| BatchedModel { size, plan: m })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { models, model_api })
    }

    /// Create an inferer for the provided typed model.
    ///
    /// # Errors
    ///
    /// Will only forward errors from the [`tract_core::model::Graph`] optimization and graph building steps.
    pub fn from_typed(model: TypedModel, sizes: &[usize]) -> TractResult<Self> {
        let model_api = ModelApi::for_typed_model(&model.clone())?;

        let sizes = fixup_sizes(sizes);

        let models = sizes
            .into_iter()
            .map(|size| {
                helpers::build_typed(model.clone(), size as i32)
                    .map(|m| BatchedModel { size, plan: m })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { models, model_api })
    }
}

impl Inferer for FixedBatchInferer {
    fn infer_batched<'pad, 'result>(
        &'result mut self,
        batch: ScratchPadView<'pad>,
    ) -> Result<BatchResponse<'result>, anyhow::Error> {
        let plan = self
            .models
            .iter_mut()
            .find(|plan| plan.size == batch.len())
            .with_context(|| anyhow::anyhow!("looking for a plan with size {:?}", batch.len()))?;

        plan.execute(batch, &self.model_api)
    }

    fn select_batch_size(&mut self, max_count: usize) -> usize {
        // Find the smallest batch size below or equal to max_count
        self.models
            .iter()
            .map(|plan| plan.size)
            .find(|size| *size <= max_count)
            .unwrap()
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
    fn build_inputs(
        &mut self,
        batch: ScratchPadView,
        model_api: &ModelApi,
    ) -> Result<TVec<Tensor>> {
        assert_eq!(batch.len(), self.size);
        let size = self.size;

        let mut inputs = TVec::default();

        for (idx, (name, shape)) in model_api.inputs.iter().enumerate() {
            assert_eq!(name, batch.slot_name(idx));

            let mut full_shape = tvec![size];
            full_shape.extend_from_slice(shape);

            let total_count: usize = full_shape.iter().product();
            assert_eq!(
                total_count,
                batch.slot(idx).len(),
                "mismatched number of features: expected {:?}, got {:?} for shape {:?}",
                total_count,
                batch.slot(idx).len(),
                full_shape
            );

            let shape = full_shape;

            let tensor = Tensor::from_shape(&shape, batch.slot(idx))?;

            inputs.push(tensor);
        }

        Ok(inputs)
    }

    fn execute<'a>(
        &mut self,
        observations: ScratchPadView,
        model_api: &'a ModelApi,
    ) -> Result<BatchResponse<'a>> {
        let inputs = self.build_inputs(observations, model_api)?;

        let result = self.plan.run(inputs)?;

        let mut output = BatchResponse::empty();
        for (idx, (name, _)) in model_api.outputs.iter().enumerate() {
            let value = result[idx]
                .to_array_view::<f32>()?
                .as_slice()
                .unwrap()
                .to_owned();

            output.insert(name, value);
        }

        Ok(output)
    }
}
