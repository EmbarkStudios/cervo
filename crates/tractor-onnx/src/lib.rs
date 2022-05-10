/// Contains utilities for interacting with ONNX specifically.
use std::io::Read;
use tract_onnx::WithOnnx;
use tractor::tract_wrap::TractInstance;

use anyhow::Result;
use tract_onnx::{prelude::*, tract_hir::infer::Factoid};

/// Create an unbatched inferer from the provided bytes reader
pub fn inferer_from_stream(reader: &mut dyn Read) -> Result<TractInstance> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_read(reader)?;

    TractInstance::from_model(model, &[1])
}

/// Create an batched inferer from the provided bytes reader
pub fn batched_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<TractInstance> {
    let onnx = tract_onnx::onnx();
    let model = onnx.model_for_read(reader)?;

    TractInstance::from_model(model, batch_size)
}

/// Convert an ONNX model to a NNEF model.
pub fn to_nnef(reader: &mut dyn Read) -> Result<Vec<u8>> {
    let onnx = tract_onnx::onnx();

    let mut model = onnx.model_for_read(reader)?;
    let batch = 1;

    let input_outlets = model.input_outlets()?.to_vec();
    for input_outlet in input_outlets {
        let input_shape = &model.input_fact(input_outlet.node)?.shape;
        let mut shape: Vec<_> = input_shape
            .dims()
            .skip(1)
            .map(|fact| fact.concretize().unwrap())
            .collect();

        shape.insert(0, batch.to_dim());
        eprintln!("shape: {:?}", shape);
        model.set_input_fact(
            input_outlet.node,
            InferenceFact::dt_shape(DatumType::F32, &shape),
        )?;
    }

    // let output_outlets = model.output_outlets()?.to_vec();
    // for output_outlet in output_outlets {
    //     let output_shape = &model.output_fact(output_outlet.node)?.shape;

    //     let mut shape: Vec<_> = output_shape
    //         .dims()
    //         .map(|fact| fact.concretize().unwrap())
    //         .collect();

    //     shape.insert(0, batch.to_dim());
    //     model.set_output_fact(
    //         output_outlet.slot,
    //         InferenceFact::dt_shape(DatumType::F32, &shape),
    //     )?;
    // }

    let mut model = model.into_typed()?;
    model.declutter()?;
    let mut output = vec![];
    let nnef = tract_nnef::nnef().with_tract_core().with_onnx();
    nnef.write(&model, &mut output)?;
    Ok(output)
}
