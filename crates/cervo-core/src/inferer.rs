// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2020, Embark Studios, all rights reserved.
// Created: 18 May 2020

#![warn(clippy::all)]

/*!
Inferers is the main access-point for cervo; providing a higher-level API on top of `tract`. Irregardless of
inferer-flavour you choose, cervo tries to provide a uniform API for batched dictionary based inference.

Using dictionary-based inference comes at a performance overhead; but helps maintain some generality. Our use-case
hasn't shown that this is significant enough to warrant the fiddliness of other approaches - interning, slot-markers -
or the fragility of delegating input-building a layer up.

## Choosing an inferer

 <p style="background:rgba(255,181,77,0.16);padding:0.75em;">
<strong>Note:</strong> Our inferer setup hs been iterated on since 2019, and we've gone through a few variants and tested a bunch of different infering setups. While important distinctions remain, it's important to know that on x86_64 platforms tract will use a kernel optimized for a batch size of 1 or 6 elements, picking whichever works best. Similar patterns exist on arm64. This is being worked on, but limits the performance gain from various batching strategies.</p>

Cervo currently provides three different inferers, two of which we've used historially (basic and fixed) and one recent addition that isn't as tested (dynamic). You'll find more detail on each page, but here comes a quick rundown of the various use cases:

| Inferer | Batch size   | Memory use                         | Performance |
| ------- | ------------ | ---------------------------------- | ----------- |
| Basic   | 1            | Fixed                              | Linear with number of elements |
| Fixed   | Known, exact | Fixed, linear with number of batch sizes | Optimal if exact match |
| Dynamic | Unknown      | Linear with number of batch sizes  | Optimal, high cost for new batch size |

As a rule of thumb, use a basic inferer if you'll almost always pass a
single item. If you need more items and know how many, use a fixed
inferer. Otherwise, use a dynamic batcher if you can afford the spikes
and potential memory use. As a final option a fixed batcher with `[1,
3, 6]` is a reasonable choice.

*/
use anyhow::{Error, Result};
use std::collections::HashMap;

mod basic;
mod dynamic;
mod fixed;
mod helpers;

pub use basic::BasicInferer;
pub use dynamic::DynamicMemoizingInferer;
pub use fixed::FixedBatchInferer;

use crate::{EpsilonInjector, NoiseGenerator};

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

/// Helper trait to provide helper functions for loadable models.
pub trait InfererProvider {
    /// Build a [`BasicInferer`] without an epsilon.
    fn build_basic(self) -> Result<BasicInferer>;

    /// Build a [`BasicInferer`] without an epsilon.
    fn build_fixed(self, sizes: &[usize]) -> Result<FixedBatchInferer>;

    /// Build a [`DynamicMemoizingInferer`] without an epsilon.
    fn build_memoizing(self, preload_sizes: &[usize]) -> Result<DynamicMemoizingInferer>;
}

/// Builder for inferers.
pub struct InfererBuilder<P: InfererProvider> {
    provider: P,
}

impl<P> InfererBuilder<P>
where
    P: InfererProvider,
{
    /// Begin the building process from the provided model provider.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Build a [`BasicInferer`] without an epsilon.
    pub fn build_basic(self) -> Result<BasicInferer> {
        self.provider.build_basic()
    }

    /// Build a [`BasicInferer`] without an epsilon.
    pub fn build_fixed(self, sizes: &[usize]) -> Result<FixedBatchInferer> {
        self.provider.build_fixed(sizes)
    }

    /// Build a [`DynamicMemoizingInferer`] without an epsilon.
    pub fn build_memoizing(self, preload_sizes: &[usize]) -> Result<DynamicMemoizingInferer> {
        self.provider.build_memoizing(preload_sizes)
    }
}

pub trait InfererExt: Inferer + Sized {
    /// Add an epsilon injector using the default noise kind.
    fn with_default_epsilon(self, key: &str) -> Result<EpsilonInjector<Self>> {
        EpsilonInjector::wrap(self, key)
    }

    /// Add an epsilon injector with a specific noise generator.
    fn with_epsilon<G: NoiseGenerator>(
        self,
        generator: G,
        key: &str,
    ) -> Result<EpsilonInjector<Self, G>> {
        EpsilonInjector::with_generator(self, generator, key)
    }
}

impl<T> InfererExt for T where T: Inferer + Sized {}
