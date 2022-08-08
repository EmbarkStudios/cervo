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

 <p style="background:rgba(255,181,77,0.16);padding:0.75em;"> <strong>Note:</strong> Our inferer setup has been iterated
on since 2019, and we've gone through a few variants and tested a bunch of different infering setups. See the rule of
thumb for selecting an inferer below; but it is suggested to benchmark. While undocumented, you can use the code in the
`perf-test` folder on GitHub to run various benchmarks.</p>

Cervo currently provides four different inferers, two of which we've used historially (basic and fixed) and two based on
newer tract functionalities that we've not tested as much yet. You'll find more detail on each page, but here comes a
quick rundown of the various use cases:

| Inferer   | Batch size   | Memory use                                          | Performance |
| --------- | ------------ | --------------------------------------------------- | ----------- |
| Basic     | 1            | Fixed                                               | Linear with number of elements |
| Fixed     | Known, exact | Fixed, linear with number of configured batch sizes | Optimal if exact match                |
| Memoizing | Unknown      | Linear with number of batch sizes                   | Optimal, high cost for new batch size |
| Dynamic   | Unknown      | Fixed                                               | Good scaling but high overhead         |

As a rule of thumb, use a basic inferer if you'll almost always pass a single item. If you need more items and know how
many, use a fixed inferer. Otherwise, use a memoizing inferer if you can afford the spikes and potential memory use. As
a final resort you can use the true dynamic inferer trading off the memory use for worse performance.
 */

use anyhow::{Error, Result};
use std::collections::HashMap;

mod basic;
mod dynamic;
mod fixed;
mod helpers;
mod memoizing;

pub use basic::BasicInferer;
pub use dynamic::DynamicInferer;
pub use fixed::FixedBatchInferer;
pub use memoizing::MemoizingDynamicInferer;

use crate::{
    batcher::{Batched, Batcher, ScratchPadView},
    epsilon::{EpsilonInjector, NoiseGenerator},
};

/// The data of one element in a batch.
#[derive(Clone, Debug)]
pub struct State<'a> {
    pub data: HashMap<&'a str, Vec<f32>>,
}

impl<'a> State<'a> {
    /// Create a new empty state to fill with data
    pub fn empty() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

/// The output for one batch element.
#[derive(Clone, Debug, Default)]
pub struct Response<'a> {
    pub data: HashMap<&'a str, Vec<f32>>,
}

impl<'a> Response<'a> {
    /// Create a new empty state to fill with data
    pub fn empty() -> Self {
        Self {
            data: Default::default(),
        }
    }

    pub fn append(&mut self, other: Response<'_>) {
        for (k, v) in other.data {
            self.data.get_mut(k).unwrap().extend_from_slice(&v);
        }
    }
}

/// The main workhorse shared by all components in Cervo.
pub trait Inferer {
    /// Query the inferer for how many elements it can deal with in a single batch.
    fn select_batch_size(&self, max_count: usize) -> usize;

    /// Execute the model on the provided pre-batched data.
    fn infer_raw(&mut self, batch: ScratchPadView) -> Result<(), anyhow::Error>;

    /// Retrieve the name and shapes of the model inputs.
    fn input_shapes(&self) -> &[(String, Vec<usize>)];

    /// Retrieve the name and shapes of the model outputs.
    fn output_shapes(&self) -> &[(String, Vec<usize>)];
}

/// Helper trait to provide helper functions for loadable models.
pub trait InfererProvider {
    /// Build a [`BasicInferer`].
    fn build_basic(self) -> Result<BasicInferer>;

    /// Build a [`FixedBatchInferer`].
    fn build_fixed(self, sizes: &[usize]) -> Result<FixedBatchInferer>;

    /// Build a [`MemoizingDynamicInferer`].
    fn build_memoizing(self, preload_sizes: &[usize]) -> Result<MemoizingDynamicInferer>;

    /// Build a [`DynamicInferer`].
    fn build_dynamic(self) -> Result<DynamicInferer>;
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

    /// Build a [`BasicInferer`].
    pub fn build_basic(self) -> Result<BasicInferer> {
        self.provider.build_basic()
    }

    /// Build a [`FixedBatchInferer`].
    pub fn build_fixed(self, sizes: &[usize]) -> Result<FixedBatchInferer> {
        self.provider.build_fixed(sizes)
    }

    /// Build a [`DynamicInferer`].
    pub fn build_dynamic(self) -> Result<DynamicInferer> {
        self.provider.build_dynamic()
    }

    /// Build a [`MemoizingDynamicInferer`].
    pub fn build_memoizing(self, preload_sizes: &[usize]) -> Result<MemoizingDynamicInferer> {
        self.provider.build_memoizing(preload_sizes)
    }
}

/// Extension trait for [`Inferer`].
// TODO[TSolberg]: This was intended to be part of the builder but it becomes an awful state-machine and is hard to extend.
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

    /// Wrap in a batching interface.
    fn into_batched(self) -> Batched<Self> {
        Batched::wrap(self)
    }

    /// Execute the model on the provided batch of elements.
    #[deprecated(
        note = "Please use the more explicit 'infer_batch' instead.",
        since = "0.3.0"
    )]
    fn infer(
        &mut self,
        observations: HashMap<u64, State>,
    ) -> Result<HashMap<u64, Response>, Error> {
        self.infer_batch(observations)
    }

    /// Execute the model on the provided pre-batched data.
    fn infer_batch<'this>(
        &'this mut self,
        batch: HashMap<u64, State>,
    ) -> Result<HashMap<u64, Response<'this>>, anyhow::Error> {
        let mut batcher = Batcher::new_sized(self, batch.len());
        batcher.extend(batch)?;

        batcher.execute(self)
    }

    /// Execute the model on the provided pre-batched data.
    fn infer_single<'this>(&'this mut self, input: State) -> Result<Response<'this>, anyhow::Error>
    where
        Self: Sized,
    {
        let mut batcher = Batcher::new_sized(self, 1);
        batcher.push(0, input)?;

        Ok(batcher.execute(self)?.remove(&0).unwrap())
    }
}

impl<T> InfererExt for T where T: Inferer + Sized {}

impl Inferer for Box<dyn Inferer> {
    fn select_batch_size(&self, max_count: usize) -> usize {
        self.as_ref().select_batch_size(max_count)
    }

    fn infer_raw(&mut self, batch: ScratchPadView) -> Result<(), anyhow::Error> {
        self.as_mut().infer_raw(batch)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.as_ref().input_shapes()
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.as_ref().output_shapes()
    }
}
