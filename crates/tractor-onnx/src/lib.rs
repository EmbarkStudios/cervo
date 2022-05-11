/// Contains utilities for interacting with ONNX specifically.
use std::io::Read;
use tract_onnx::WithOnnx;
use tractor::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

use anyhow::Result;
use tract_onnx::{prelude::*, tract_hir::infer::Factoid};

/// Create a basic inferer from the provided bytes reader.
///
/// See [`BasicInferer`] for more details.
pub fn simple_inferer_from_stream(reader: &mut dyn Read) -> Result<BasicInferer> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_read(reader)?;

    BasicInferer::from_model(model)
}

/// Create an dynamic batching inferer from the provided bytes reader
///
/// See [`DynamicBatchingInferer`] for more details.
pub fn batched_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<DynamicBatchingInferer> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_read(reader)?;

    DynamicBatchingInferer::from_model(model, batch_size)
}

/// Create an fixed batching inferer from the provided bytes reader
///
/// See [`DynamicBatchingInferer`] for more details.
pub fn fixed_batch_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<FixedBatchingInferer> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_read(reader)?;

    FixedBatchingInferer::from_model(model, batch_size)
}

/// Convert an ONNX model to a NNEF model.
pub fn to_nnef(reader: &mut dyn Read) -> Result<Vec<u8>> {
    let onnx = tract_onnx::onnx();

    let mut model = onnx.model_for_read(reader)?;
    let batch = Symbol::new('N');

    let input_outlets = model.input_outlets()?.to_vec();
    for input_outlet in input_outlets {
        let input_shape = &model.input_fact(input_outlet.node)?.shape;
        let mut shape: Vec<_> = input_shape
            .dims()
            .skip(1)
            .map(|fact| fact.concretize().unwrap())
            .collect();

        shape.insert(0, batch.to_dim());

        model.set_input_fact(
            input_outlet.node,
            InferenceFact::dt_shape(DatumType::F32, &shape),
        )?;
    }

    let mut model = model.into_typed()?;
    model.declutter()?;
    let mut output = vec![];
    let nnef = tract_nnef::nnef().with_tract_core().with_onnx();
    nnef.write(&model, &mut output)?;
    Ok(output)
}
