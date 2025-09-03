// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 11 May 2022

/*!
Utilities for filling noise inputs for an inference model.
*/

use std::cell::RefCell;

use crate::{batcher::ScratchPadView, inferer::Inferer, prelude::ModelWrapper};
use anyhow::{bail, Result};
use perchance::PerchanceContext;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

/// `NoiseGenerators` are consumed by the [`EpsilonInjector`] by generating noise sampled for a standard normal
/// distribution. Custom noise-generators can be implemented and passed via [`EpsilonInjector::with_generator`].
pub trait NoiseGenerator {
    fn generate(&self, count: usize, out: &mut [f32]);
}

/// A non-noisy noise generator, primarily intended for debugging or testing purposes.
pub struct ConstantGenerator {
    value: f32,
}

impl ConstantGenerator {
    /// Will generate the provided `value` when called.
    pub fn for_value(value: f32) -> Self {
        Self { value }
    }

    /// Convenience function for a constant generator for zeros.
    pub fn zeros() -> Self {
        Self::for_value(1.0)
    }

    /// Convenience function for a constant generator for ones.
    pub fn ones() -> Self {
        Self::for_value(1.0)
    }
}

impl NoiseGenerator for ConstantGenerator {
    fn generate(&self, _count: usize, out: &mut [f32]) {
        for o in out {
            *o = self.value;
        }
    }
}

/// A low quality noise generator which is about twice as fast as the built-in [`HighQualityNoiseGenerator`]. This uses
/// an XORSHIFT algorithm internally which isn't cryptographically secure.
///
/// The default implementation will seed the generator from the current time.
pub struct LowQualityNoiseGenerator {
    ctx: RefCell<PerchanceContext>,
}

impl LowQualityNoiseGenerator {
    /// Create a new LQNG with the provided fixed seed.
    pub fn new(seed: u128) -> Self {
        Self {
            ctx: RefCell::new(PerchanceContext::new(seed)),
        }
    }
}

impl Default for LowQualityNoiseGenerator {
    fn default() -> Self {
        Self {
            ctx: RefCell::new(PerchanceContext::new(perchance::gen_time_seed())),
        }
    }
}

impl NoiseGenerator for LowQualityNoiseGenerator {
    /// Generate `count` random values.
    fn generate(&self, _count: usize, out: &mut [f32]) {
        let mut ctx = self.ctx.borrow_mut();
        for o in out {
            *o = ctx.normal_f32();
        }
    }
}

/// A high quality noise generator which is measurably slower than the LQGN, but still fast enough for most real-time
/// use-cases.
///
/// This implementation uses [`rand::thread_rng`] internally as the entropy source, and uses the optimized
/// `StandardNormal` distribution for sampling.
pub struct HighQualityNoiseGenerator {
    normal_distribution: StandardNormal,
}

impl Default for HighQualityNoiseGenerator {
    fn default() -> Self {
        Self {
            normal_distribution: StandardNormal,
        }
    }
}

impl NoiseGenerator for HighQualityNoiseGenerator {
    /// Generate `count` random values.
    fn generate(&self, _count: usize, out: &mut [f32]) {
        let mut rng = thread_rng();
        for o in out {
            *o = self.normal_distribution.sample(&mut rng);
        }
    }
}

struct EpsilonInjectorState<NG: NoiseGenerator> {
    count: usize,
    index: usize,
    generator: NG,

    inputs: Vec<(String, Vec<usize>)>,
}
/// The [`EpsilonInjector`] wraps an inferer to add noise values as one of the input data points. This is useful for
/// continuous action policies where you might have trained your agent to follow a stochastic policy trained with the
/// reparametrization trick.
///
/// Note that it's fully possible to pass an epsilon directly in your observation, and this is purely a convenience
/// wrapper.
pub struct EpsilonInjector<T: Inferer, NG: NoiseGenerator = HighQualityNoiseGenerator> {
    inner: T,

    state: EpsilonInjectorState<NG>,
}

impl<T> EpsilonInjector<T, HighQualityNoiseGenerator>
where
    T: Inferer,
{
    /// Wraps the provided `inferer` to automatically generate noise for the input named by `key`.
    ///
    /// This function will use [`HighQualityNoiseGenerator`] as the noise source.
    ///
    /// # Errors
    ///
    /// Will return an error if the provided key doesn't match an input on the model.
    pub fn wrap(inferer: T, key: &str) -> Result<EpsilonInjector<T, HighQualityNoiseGenerator>> {
        Self::with_generator(inferer, HighQualityNoiseGenerator::default(), key)
    }
}

impl<T, NG> EpsilonInjector<T, NG>
where
    T: Inferer,
    NG: NoiseGenerator,
{
    /// Create a new injector for the provided `key`, using the custom `generator` as the noise source.
    ///
    /// # Errors
    ///
    /// Will return an error if the provided key doesn't match an input on the model.
    pub fn with_generator(inferer: T, generator: NG, key: &str) -> Result<Self> {
        let inputs = inferer.input_shapes();

        let (index, count) = match inputs.iter().enumerate().find(|(_, (k, _))| k == key) {
            Some((index, (_, shape))) => (index, shape.iter().product()),
            None => bail!("model has no input key {:?}", key),
        };

        let inputs = inputs
            .iter()
            .filter(|(k, _)| *k != key)
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect::<Vec<_>>();

        Ok(Self {
            inner: inferer,
            state: EpsilonInjectorState {
                index,
                count,
                generator,
                inputs,
            },
        })
    }
}

impl<T, NG> Inferer for EpsilonInjector<T, NG>
where
    T: Inferer,
    NG: NoiseGenerator,
{
    fn select_batch_size(&self, max_count: usize) -> usize {
        self.inner.select_batch_size(max_count)
    }

    fn infer_raw(&self, batch: &mut ScratchPadView<'_>) -> Result<(), anyhow::Error> {
        let total_count = self.state.count * batch.len();
        let output = batch.input_slot_mut(self.state.index);
        self.state.generator.generate(total_count, output);

        self.inner.infer_raw(batch)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.state.inputs
    }

    fn raw_input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.raw_input_shapes()
    }

    fn raw_output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.raw_output_shapes()
    }

    fn begin_agent(&mut self, id: u64) {
        self.inner.begin_agent(id);
    }

    fn end_agent(&mut self, id: u64) {
        self.inner.end_agent(id);
    }
}

pub struct EpsilonInjectorWrapper<Inner: ModelWrapper, NG: NoiseGenerator> {
    inner: Inner,
    state: EpsilonInjectorState<NG>,
}

impl<Inner: ModelWrapper> EpsilonInjectorWrapper<Inner, HighQualityNoiseGenerator> {
    /// Wraps the provided `inferer` to automatically generate noise for the input named by `key`.
    ///
    /// This function will use [`HighQualityNoiseGenerator`] as the noise source.
    ///
    /// # Errors
    ///
    /// Will return an error if the provided key doesn't match an input on the model.
    pub fn wrap(
        inner: Inner,
        inferer: &dyn Inferer,
        key: &str,
    ) -> Result<EpsilonInjectorWrapper<Inner, HighQualityNoiseGenerator>> {
        Self::with_generator(inner, inferer, HighQualityNoiseGenerator::default(), key)
    }
}

impl<Inner, NG> EpsilonInjectorWrapper<Inner, NG>
where
    Inner: ModelWrapper,
    NG: NoiseGenerator,
{
    /// Create a new injector for the provided `key`, using the custom `generator` as the noise source.
    ///
    /// # Errors
    ///
    /// Will return an error if the provided key doesn't match an input on the model.
    pub fn with_generator(
        inner: Inner,
        inferer: &dyn Inferer,
        generator: NG,
        key: &str,
    ) -> Result<Self> {
        let inputs = inferer.input_shapes();

        let (index, count) = match inputs.iter().enumerate().find(|(_, (k, _))| k == key) {
            Some((index, (_, shape))) => (index, shape.iter().product()),
            None => bail!("model has no input key {:?}", key),
        };

        let inputs = inputs
            .iter()
            .filter(|(k, _)| *k != key)
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect::<Vec<_>>();

        Ok(Self {
            inner,
            state: EpsilonInjectorState {
                index,
                count,
                generator,
                inputs,
            },
        })
    }
}

impl<Inner, NG> ModelWrapper for EpsilonInjectorWrapper<Inner, NG>
where
    Inner: ModelWrapper,
    NG: NoiseGenerator,
{
    fn invoke(&self, inferer: &impl Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()> {
        self.inner.invoke(inferer, batch)?;
        let total_count = self.state.count * batch.len();
        let output = batch.input_slot_mut(self.state.index);
        self.state.generator.generate(total_count, output);

        self.inner.invoke(inferer, batch)
    }

    fn input_shapes<'a>(&'a self, _inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        self.state.inputs.as_ref()
    }

    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        self.inner.output_shapes(inferer)
    }

    fn begin_agent(&self, id: u64) {
        self.inner.begin_agent(id)
    }

    fn end_agent(&self, id: u64) {
        self.inner.end_agent(id)
    }
}
