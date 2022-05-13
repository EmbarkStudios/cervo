/// Contains utilities for using tractor with ONNX.
use anyhow::Result;
use std::cell::UnsafeCell;
use std::io::Read;
use std::rc::Rc;
use tract_nnef::{framework::Nnef, prelude::*};

use tractor::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

thread_local!(
    static NNEF: Rc<UnsafeCell<Nnef>>  = {

        Rc::new(UnsafeCell::new(tract_nnef::nnef().with_tract_core()))
    }
);

pub fn model_for_reader(reader: &mut dyn Read) -> Result<TypedModel> {
    NNEF.with(|n| unsafe { (&*n.as_ref().get()).model_for_read(reader) })
}

/// Create a basic inferer from the provided bytes reader.
///
/// See [`BasicInferer`] for more details.
pub fn simple_inferer_from_stream(reader: &mut dyn Read) -> Result<BasicInferer> {
    let model = model_for_reader(reader)?;
    BasicInferer::from_typed(model)
}

/// Create an dynamic batching inferer from the provided bytes reader
///
/// See [`DynamicBatchingInferer`] for more details.
pub fn batched_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<DynamicBatchingInferer> {
    let model = model_for_reader(reader)?;
    DynamicBatchingInferer::from_typed(model, batch_size)
}

/// Create an fixed batching inferer from the provided bytes reader
///
/// See [`FixedBatchingInferer`] for more details.
pub fn fixed_batch_inferer_from_stream(
    reader: &mut dyn Read,
    batch_size: &[usize],
) -> Result<FixedBatchingInferer> {
    let model = model_for_reader(reader)?;
    FixedBatchingInferer::from_typed(model, batch_size)
}
