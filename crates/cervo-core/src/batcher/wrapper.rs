// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 27 July 2022

use super::Batcher;
use crate::inferer::{Inferer, Response, State};
use std::collections::HashMap;

/// Wraps an inferer in a batching interface. This'll separate the
/// data-insertion and execution, which generally improves
/// performance.
///
/// Can be easily constructed using [InfererExt::into_batched](crate::prelude::InfererExt::into_batched).
pub struct Batched<Inf: Inferer> {
    inner: Inf,
    batcher: Batcher,
}

impl<Inf> Batched<Inf>
where
    Inf: Inferer,
{
    /// Wrap the provided inferer.
    pub fn wrap(inferer: Inf) -> Self {
        let batcher = Batcher::new(&inferer);
        Self {
            batcher,
            inner: inferer,
        }
    }

    /// Insert a single element into the batch to include in the next execution.
    pub fn push(&mut self, id: u64, state: State<'_>) -> anyhow::Result<()> {
        self.batcher.push(id, state)
    }

    /// Insert a sequence of elements into the batch to include in the next execution.
    pub fn extend<'a, Iter: IntoIterator<Item = (u64, State<'a>)>>(
        &mut self,
        states: Iter,
    ) -> anyhow::Result<()> {
        self.batcher.extend(states)
    }

    /// Execute the model on the data that has been enqueued previously.
    pub fn execute(&mut self) -> anyhow::Result<HashMap<u64, Response>> {
        self.batcher.execute(&self.inner)
    }

    /// Split the batcher and the inferer.
    pub fn into_parts(self) -> (Inf, Batcher) {
        (self.inner, self.batcher)
    }
}
