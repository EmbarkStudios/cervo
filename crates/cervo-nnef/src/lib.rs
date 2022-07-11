/*! Contains utilities for using cervo with NNEF.

If you're going to defer loading NNEF files to runtime, consider
running [`init`] ahead of time to remove some overhead from the first
load call.


## Loading an inference model
```no_run
# fn load_bytes(s: &str) -> std::io::Cursor<Vec<u8>> { std::io::Cursor::new(vec![]) }
use cervo_core::prelude::InfererExt;

let model_data = load_bytes("model.nnef");
let model = cervo_nnef::builder(model_data)
    .build_fixed(&[2])?
    .with_default_epsilon("epsilon");
# Ok::<(), Box<dyn std::error::Error>>(())
```

*/

use anyhow::Result;
use cervo_core::prelude::{
    BasicInferer, DynamicInferer, FixedBatchInferer, InfererBuilder, InfererProvider,
    MemoizingDynamicInferer,
};
use std::{
    ffi::OsStr,
    io::Read,
    path::{Path, PathBuf},
};
use tract_nnef::{framework::Nnef, prelude::*};

lazy_static::lazy_static! {
    static ref NNEF: Nnef  = {
        tract_nnef::nnef().with_tract_core()
    };
}

/// Initialize the global NNEF instance.
///
/// To ensure fast loading cervo uses a shared instance of the
/// tract NNEF framework. If you don't want to pay for initialization
/// on first-time load you can call this earlier to ensure it's set up
/// ahead of time.
pub fn init() {
    use lazy_static::LazyStatic;
    NNEF::initialize(&NNEF)
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
    NNEF.model_for_read(reader)
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

    /// Build a [`MemoizingDynamicInferer`].
    fn build_memoizing(mut self, preload_sizes: &[usize]) -> Result<MemoizingDynamicInferer> {
        let model = self.load()?;
        MemoizingDynamicInferer::from_typed(model, preload_sizes)
    }

    /// Build a [`DynamicInferer`].
    fn build_dynamic(mut self) -> Result<DynamicInferer> {
        let model = self.load()?;
        DynamicInferer::from_typed(model)
    }
}

/// Utility function for creating an [`InfererBuilder`] for [`NnefData`].
pub fn builder<T: Read>(read: T) -> InfererBuilder<NnefData<T>> {
    InfererBuilder::new(NnefData(read))
}
