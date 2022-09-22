// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 28 July 2022

/*!
The Cervo runtime is a combined front for multiple inference models,
managing data routing and offering higher-level constructs for
execution.

The goal of the runtime is to simplify batched execution, especially
in the face of time-slotted execution.

```no_run
use cervo_asset::{AssetData, AssetKind};
use cervo_runtime::{BrainId, Runtime, AgentId};
use std::time::Duration;
# fn load_bytes(s: &str) -> Vec<u8> { vec![] }
# fn load_model(name: &str) -> AssetData { AssetData::new(AssetKind::Onnx, load_bytes(name)) }
# mod game {
#     use std::collections::HashMap;
#     use cervo_runtime::{BrainId, Runtime, AgentId};
#     pub fn observe_trucks(key: BrainId, r: &mut Runtime) {}
#     pub fn observe_racecars(key: BrainId, r: &mut Runtime) {}
#     pub fn assign_actions(response: HashMap<BrainId, HashMap<AgentId, cervo_core::prelude::Response<'_>>>) {}
# }

let mut runtime = Runtime::new();

let racer_asset = load_model("racing-car");
let racer_inferer = racer_asset.load_basic()?;
let racer_infer_key = runtime.add_inferer(racer_inferer);

let truck_asset = load_model("monster-truck");
let truck_inferer = truck_asset.load_basic()?;
let truck_infer_key = runtime.add_inferer(truck_inferer);

game::observe_racecars(racer_infer_key, &mut runtime);
game::observe_trucks(truck_infer_key, &mut runtime);

let responses = runtime.run_for(Duration::from_millis(1))?;
game::assign_actions(responses);

# Ok::<(), Box<dyn std::error::Error>>(())
```

## Notes on time-slotted execution

There is a very experimental feature for using time-slotted execution
based on collected performance metrics. Each time a model is executed
the runtime records the batch size and time-cost. This is then used to
estimate the cost of future batches.

The implementation currently requires that whichever model is first in
the queue for execution gets to run. This is to ensure that models
don't end up in back-off. This means that
`Runtime::run_for(Duration::from_millis(0))` will still execute one
model.

Once the first model has executed, models will be processed in order,
starting by the one that has waited longest for running. Models that
would take too long to run will be skipped and end up at the back of
the queue. This ensures that the first skipped model is at the start
of the queue next round.

The estimation algorithm uses Welford's Online Algorithm which can
integrate mean and variance without requiring extra storage. However,
this update method can be quite unstable with few samples. This can
lead to some stuttering early on by underestimating cost, or running
too few models by overestimation.

 */

#![warn(rust_2018_idioms)]

mod error;
mod runtime;
mod state;
mod timing;

#[doc(inline)]
pub use crate::error::CervoError;
#[doc(inline)]
pub use runtime::Runtime;

/// Identifier for a specific brain.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct BrainId(pub u16);

/// Identifier for a specific agent.
pub type AgentId = u64;
