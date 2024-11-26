/*!

# Cervo

Cervo is a toolkit we use for ML integration in games. The core
use-case is for RL-agents, and some utilities for managing brain assets.

## Cervo Core

The core crate focuses on wrappers for tract models. It adds a few
different modes for running inferers, as well as data injectors for
stochastic policies (e.g. SAC).

```no_run
# fn load_bytes(s: &str) -> std::io::Cursor<Vec<u8>> { std::io::Cursor::new(vec![]) }
# use cervo_onnx::tract_onnx;
# use cervo_onnx::tract_onnx::prelude::*;
use cervo_core::prelude::{BasicInferer, InfererExt};

let mut model_data = load_bytes("model.onnx");
let inference_model = tract_onnx::onnx().model_for_read(&mut model_data)?;

let inferer = BasicInferer::from_model(inference_model)?
    .with_default_epsilon("noise");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Cervo Asset

To support isomg NNEF and ONNX interchangeably we have a small
wrapping binary format which can contain either type of data, helping
keep track of which data is what.

```no_run
# fn load_bytes(s: &str) -> Vec<u8> { vec![] }
use cervo_asset::{AssetData, AssetKind};

let model_data = load_bytes("model.onnx");
let asset = AssetData::new(AssetKind::Onnx, model_data);

// convert to a symbolic NNEF asset, with deterministic timestamps
let nnef_asset = asset.to_nnef(None, true)?;

let inferer = asset.load_basic();
let nnef_inferer = nnef_asset.load_fixed(&[42]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Cervo ONNX and Cervo NNEF

These are simple intermediates helping Cervo Asset, but can also be used directly.

```no_run
# fn load_bytes(s: &str) -> std::io::Cursor<Vec<u8>> { std::io::Cursor::new(vec![]) }
use cervo_core::prelude::InfererExt;

let model_data = load_bytes("model.onnx");
let model = cervo_onnx::builder(model_data)
    .build_memoizing(&[])?
    .with_default_epsilon("epsilon");
# Ok::<(), Box<dyn std::error::Error>>(())
```


*/

#![warn(rust_2018_idioms)]

pub use cervo_asset as asset;
pub use cervo_core as core;
pub use cervo_nnef as nnef;
pub use cervo_onnx as onnx;
pub use cervo_runtime as runtime;
