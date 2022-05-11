/// Contains utilities for using tractor with ONNX.
use anyhow::Result;
use std::io::Read;
use tract_onnx::WithOnnx;
use tract_onnx::{prelude::*, tract_hir::infer::Factoid};
use tractor::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

fn model_for_reader(reader: &mut dyn Read) -> Result<InferenceModel> {
    let onnx = tract_onnx::onnx();
    onnx.model_for_read(reader)
}

/// Create a basic inferer from the provided bytes reader.
///
/// See [`BasicInferer`] for more details.
pub fn simple_inferer_from_stream(reader: &mut dyn Read) -> Result<BasicInferer> {
    let model = model_for_reader(reader)?;
    BasicInferer::from_model(model)
}

/// Create an dynamic batching inferer from the provided bytes reader
///
/// See [`DynamicBatchingInferer`] for more details.
pub fn batched_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<DynamicBatchingInferer> {
    let model = model_for_reader(reader)?;
    DynamicBatchingInferer::from_model(model, batch_size)
}

/// Create an fixed batching inferer from the provided bytes reader
///
/// See [`FixedBatchingInferer`] for more details.
pub fn fixed_batch_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<FixedBatchingInferer> {
    let model = model_for_reader(reader)?;
    FixedBatchingInferer::from_model(model, batch_size)
}
