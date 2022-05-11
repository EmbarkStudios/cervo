// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2020, Embark Studios, all rights reserved.
// Created: 18 May 2020

#![warn(clippy::all)]

/**
Inferers is the main access-point for tractor; providing a higher-level API on top of [`tract`]. Irregardless of
inferer-flavour you choose, tractor tries to provide a uniform API for batched dictionary based inference.

Using dictionary-based inference comes at a performance overhead; but helps maintain some generality. Our use-case
hasn't shown that this is significant enough to warrant the fiddliness of other approaches - interning, slot-markers -
or the fragility of delegating input-building a layer up.
*/
use anyhow::Error;
use std::collections::HashMap;

mod basic;
mod dynamic;
mod fixed;

pub use basic::BasicInferer;
pub use dynamic::DynamicBatchingInferer;
pub use fixed::FixedBatchingInferer;

/// The data of one element in a batch.
#[derive(Clone, Debug)]
pub struct State {
    pub data: HashMap<String, Vec<f32>>,
}

/// The output for one batch element.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub data: HashMap<String, Vec<f32>>,
}

/// The main workhorse shared by all components in Tractor.
pub trait Inferer {
    /// Execute the model on the provided batch of elements.
    fn infer(&mut self, observations: HashMap<u64, State>)
        -> Result<HashMap<u64, Response>, Error>;

    /// Retrieve the name and shapes of the model inputs.
    fn input_shapes(&self) -> &[(String, Vec<usize>)];

    /// Retrieve the name and shapes of the model outputs.
    fn output_shapes(&self) -> &[(String, Vec<usize>)];
}
