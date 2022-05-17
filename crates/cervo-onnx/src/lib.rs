/// Contains utilities for using cervo with ONNX.
use anyhow::Result;
use cervo_core::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};
use std::io::Read;
use tract_onnx::{prelude::*, tract_hir::infer::Factoid};

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
