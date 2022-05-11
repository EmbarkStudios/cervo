// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 11 May 2022

/*!

*/

use std::collections::HashMap;

use crate::inferer::{Inferer, Observation, Response};
use anyhow::{bail, Result};
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

use perchance::PerchanceContext;

pub trait NoiseGenerator: Default {
    fn generate(&mut self, count: usize) -> Vec<f32>;
}

pub struct LowQualityNoiseGenerator {
    ctx: PerchanceContext,
}

impl Default for LowQualityNoiseGenerator {
    fn default() -> Self {
        Self {
            ctx: PerchanceContext::new(perchance::gen_time_seed()),
        }
    }
}

impl NoiseGenerator for LowQualityNoiseGenerator {
    fn generate(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.ctx.normal_f32()).collect()
    }
}

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
    fn generate(&mut self, count: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| self.normal_distribution.sample(&mut rng))
            .collect()
    }
}

pub struct EpsilonInjector<T: Inferer, NG: NoiseGenerator = HighQualityNoiseGenerator> {
    inner: T,
    key: String,
    count: usize,

    generator: NG,
}

impl<T> EpsilonInjector<T, HighQualityNoiseGenerator>
where
    T: Inferer,
{
    pub fn wrap(inferer: T, key: &str) -> Result<EpsilonInjector<T, HighQualityNoiseGenerator>> {
        let inputs = inferer.input_shapes();

        let count = match inputs.iter().find(|(k, _)| k == key) {
            Some((_, shape)) => shape.iter().product(),
            None => bail!("model has no input key {:?}", key),
        };

        Ok(Self {
            inner: inferer,
            key: key.to_owned(),
            count,
            generator: HighQualityNoiseGenerator::default(),
        })
    }
}

impl<T, NG> EpsilonInjector<T, NG>
where
    T: Inferer,
    NG: NoiseGenerator,
{
    pub fn with_generator(inferer: T, generator: NG, key: &str) -> Result<Self> {
        let inputs = inferer.input_shapes();

        let count = match inputs.iter().find(|(k, _)| k == key) {
            Some((_, shape)) => shape.iter().product(),
            None => bail!("model has no input key {:?}", key),
        };

        Ok(Self {
            inner: inferer,
            key: key.to_owned(),
            count,
            generator,
        })
    }
}

impl<T, NG> EpsilonInjector<T, NG>
where
    T: Inferer,
    NG: NoiseGenerator,
{
    fn inject_epsilons(&mut self, observations: &mut HashMap<u64, Observation>) -> Result<()> {
        for v in observations.values_mut() {
            v.data
                .insert(self.key.clone(), self.generator.generate(self.count));
        }
        Ok(())
    }
}

impl<T, NG> Inferer for EpsilonInjector<T, NG>
where
    T: Inferer,
    NG: NoiseGenerator,
{
    fn infer(
        &mut self,
        mut observations: HashMap<u64, Observation>,
    ) -> Result<HashMap<u64, Response>> {
        self.inject_epsilons(&mut observations)?;
        self.inner.infer(observations)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.input_shapes()
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.output_shapes()
    }
}
