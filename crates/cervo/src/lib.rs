/*!

# Cervo

Cervo is a toolkit we use for ML integration in games. The core
use-case is for RL-agents, and some utilities for managing brain assets.

## Cervo Core

The core crate focuses on wrappers for tract models. It adds a few
different modes for running inferers, as well as data injectors for
stochastic policies (e.g. SAC).

```skip
use cervo_core::prelude::{BasicInferer, InfererExt};

let model_data = load_bytes("model.onnx")?;
let inference_model = tract_onnx::model_for_reader(model_data)?;

let inferer = BasicInferer::from_model(inference_model)?
    .with_default_epsilon("noise");
```

## Cervo Asset

To support isomg NNEF and ONNX interchangeably we have a small
wrapping binary format which can contain either type of data, helping
keep track of which data is what.

```skip
use cervo_asset::{AssetData, AssetKind};

let model_data = load_bytes("model.onnx")?;
let asset = AssetData(AssetKind::Onnx, model_data)?;

let nnef_asset = asset.to_nnef(None);    // convert to a symbolic NNEF asset

let inferer = asset.load_basic();
let nnef_inferer = asset.load_fixed(&[42]);
```

## Cervo ONNX and Cervo NNEF

These are simple intermediates helping Cervo Asset, but can also be used directly.

```skip
use cervo_core::prelude::InfererExt;

let model_data = load_bytes("model.onnx")?;
let model = cervo_onnx::builder(model_data)
    .build_memoizing()?
    .with_default_epsilon("epsilon");
```


*/

pub use cervo_asset as asset;
pub use cervo_core as core;
pub use cervo_nnef as nnef;
pub use cervo_onnx as onnx;
