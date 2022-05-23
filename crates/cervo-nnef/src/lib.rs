/*! Contains utilities for using cervo with NNEF.

If you're going to load NNEF files on a thread; consider using
`init_thread` when creating it - otherwise the first NNEF asset will
cause a noticeable spike.
*/

use anyhow::Result;

use cervo_core::prelude::{
    BasicInferer, DynamicMemoizingInferer, FixedBatchInferer, InfererBuilder, InfererProvider,
};
use std::{
    cell::UnsafeCell,
    ffi::OsStr,
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
};
use tract_nnef::{framework::Nnef, prelude::*};

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

/// Utility function to check if a file name is `.nnef.tar`.
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

fn model_for_reader(reader: &mut dyn Read) -> Result<TypedModel> {
    NNEF.with(|n| unsafe { (&*n.as_ref().get()).model_for_read(reader) })
}

/// A reader for providing NNEF data.
pub struct NnefData<T: Read>(pub T);

impl<T> NnefData<T>
where
    T: Read,
{
    fn load(&mut self) -> Result<TypedModel> {
        model_for_reader(&mut self.0)
    }
}

impl<T> InfererProvider for NnefData<T>
where
    T: Read,
{
    /// Build a [`BasicInferer`].
    fn build_basic(mut self) -> Result<BasicInferer> {
        let model = self.load()?;
        BasicInferer::from_typed(model)
    }

    /// Build a [`BasicInferer`].
    fn build_fixed(mut self, sizes: &[usize]) -> Result<FixedBatchInferer> {
        let model = self.load()?;
        FixedBatchInferer::from_typed(model, sizes)
    }

    /// Build a [`DynamicMemoizingInferer`].
    fn build_memoizing(mut self, preload_sizes: &[usize]) -> Result<DynamicMemoizingInferer> {
        let model = self.load()?;
        DynamicMemoizingInferer::from_typed(model, preload_sizes)
    }
}

/// Utility function for creating an [`InfererBuilder`] for [`NnefData`].
pub fn builder<T: Read>(read: T) -> InfererBuilder<NnefData<T>> {
    InfererBuilder::new(NnefData(read))
}
