//! Contains utilities for using cervo with ONNX.

use anyhow::Result;
use cervo_core::prelude::{
    BasicInferer, DynamicMemoizingInferer, FixedBatchInferer, {InfererBuilder, InfererProvider},
};
use std::io::Read;
use tract_onnx::{prelude::*, tract_hir::infer::Factoid};

fn model_for_reader(reader: &mut dyn Read) -> Result<InferenceModel> {
    let onnx = tract_onnx::onnx();
    onnx.model_for_read(reader)
}

/// Wrapper for a reader providing ONNX data.
pub struct OnnxData<T: Read>(pub T);

impl<T> OnnxData<T>
where
    T: Read,
{
    fn load(&mut self) -> Result<InferenceModel> {
        model_for_reader(&mut self.0)
    }
}

impl<T> InfererProvider for OnnxData<T>
where
    T: Read,
{
    /// Build a [`BasicInferer`].
    fn build_basic(mut self) -> Result<BasicInferer> {
        let model = self.load()?;
        BasicInferer::from_model(model)
    }

    /// Build a [`BasicInferer`].
    fn build_fixed(mut self, sizes: &[usize]) -> Result<FixedBatchInferer> {
        let model = self.load()?;
        FixedBatchInferer::from_model(model, sizes)
    }

    /// Build a [`DynamicMemoizingInferer`].
    fn build_memoizing(mut self, preload_sizes: &[usize]) -> Result<DynamicMemoizingInferer> {
        let model = self.load()?;
        DynamicMemoizingInferer::from_model(model, preload_sizes)
    }
}

/// Utility function for creating an [`InfererBuilder`] for [`OnnxData`].
pub fn builder<T: Read>(read: T) -> InfererBuilder<OnnxData<T>> {
    InfererBuilder::new(OnnxData(read))
}

/// Convert an ONNX model to a NNEF model.
pub fn to_nnef(reader: &mut dyn Read, batch_size: Option<usize>) -> Result<Vec<u8>> {
    let mut model = model_for_reader(reader)?;

    let batch = batch_size
        .map(|v| v.to_dim())
        .unwrap_or_else(|| Symbol::from('N').to_dim());

    let input_outlets = model.input_outlets()?.to_vec();

    for input_outlet in input_outlets {
        let input_shape = &model.input_fact(input_outlet.node)?.shape;

        let mut shape: Vec<_> = input_shape
            .dims()
            .skip(1)
            .map(|fact| fact.concretize().unwrap())
            .collect();

        shape.insert(0, batch.clone());

        model.set_input_fact(
            input_outlet.node,
            InferenceFact::dt_shape(DatumType::F32, &shape),
        )?;
    }

    let model = model.into_typed()?.into_decluttered()?;

    let mut output = vec![];
    let nnef = tract_nnef::nnef().with_tract_core().with_onnx();

    nnef.write(&model, &mut output)?;
    Ok(output)
}
