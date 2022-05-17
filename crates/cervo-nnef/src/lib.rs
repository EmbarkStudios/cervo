/// Contains utilities for using cervo with NNEF.
use anyhow::Result;
use std::ffi::OsStr;
use std::io::Read;
use std::path::PathBuf;
use std::rc::Rc;
use std::{cell::UnsafeCell, path::Path};
use tract_nnef::{framework::Nnef, prelude::*};

use cervo_core::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

thread_local!(
    /// We create and cache the NNEF on a per-thread basis. This is noticeably expensive to create, so we ensure it only has to happen once.
    static NNEF: Rc<UnsafeCell<Nnef>>  = {
        Rc::new(UnsafeCell::new(tract_nnef::nnef().with_tract_core()))
    }
);

/// Initialize the thread-local NNEF instance.
///
/// To ensure fast loading cervo uses a thread-local instance of the
/// cervo-NNEF package. If you don't want to pay for initialization
/// on first-time load you can call this earlier to ensure it's set up
/// ahead of time.
pub fn init_thread() {
    NNEF.with(|_| {})
}

/// Utility function to check if a file is a valid NNEF file.
pub fn is_nnef_tar(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        if ext != "tar" {
            return false;
        }

        let stem = match path.file_stem().and_then(OsStr::to_str).map(PathBuf::from) {
            Some(p) => p,
            None => return false,
        };

        if let Some(ext) = stem.extension().and_then(OsStr::to_str) {
            return ext == "nnef";
        }
    }

    false
}

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
