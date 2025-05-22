// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 22 July 2022

/*!
Tools for batching and batched execution.

Batching leads to lower memory pressure by reusing data gathering
allocations, and higher performance by being able to run larger
kernels. This is especially noticeable for networks with large matrix
multiplications where the weights do not fit in the CPU cache.
*/

mod scratch;
mod wrapper;

use self::scratch::ScratchPad;
use crate::inferer::{Inferer, Response, State};
pub use scratch::ScratchPadView;
use std::collections::HashMap;
pub use wrapper::Batched;

/// Low-level batch builder to help transition from per-entity code to
/// batched inference. Consider using the [`Batched`] wrapper instead
/// to avoid tracking two objects.
///
/// Reusing this across frames will have a noticeable performance
/// impact for large model inputs or outputs, and reduce memory
/// pressure.
///
/// Note that Batchers are specific to the inferer used for
/// initialization.
pub struct Batcher {
    scratch: ScratchPad,
}

impl Batcher {
    /// Create a new batcher for the provided inferer.
    pub fn new(inferer: &dyn Inferer) -> Self {
        Self {
            scratch: ScratchPad::new_for_shapes(inferer.input_shapes(), inferer.output_shapes()),
        }
    }

    /// Create a new batcher for the provided inferer with space for the specified batch size.
    pub fn new_sized(inferer: &dyn Inferer, size: usize) -> Self {
        Self {
            scratch: ScratchPad::new_with_size(
                inferer.input_shapes(),
                inferer.output_shapes(),
                size,
            ),
        }
    }

    #[inline]
    fn input_slot(&self, name: &str) -> Option<usize> {
        self.scratch
            .inputs
            .iter()
            .position(|slot| slot.name == name)
    }

    /// Insert a single element into the batch to include in the next execution.
    pub fn push(&mut self, id: u64, state: State<'_>) -> anyhow::Result<()> {
        self.scratch.next(id);
        for (k, v) in state.data {
            let slot = self
                .input_slot(k)
                .ok_or_else(|| anyhow::anyhow!("key doesn't match an input: {:?}", k))?;

            self.scratch.push(slot, v);
        }

        Ok(())
    }

    /// Insert a sequence of elements into the batch to include in the next execution.
    pub fn extend<'a, Iter: IntoIterator<Item = (u64, State<'a>)>>(
        &mut self,
        states: Iter,
    ) -> anyhow::Result<()> {
        for (id, state) in states {
            self.push(id, state)?;
        }

        Ok(())
    }

    /// Run the provided inferer on the data that has been enqueued previously.
    pub fn execute<'b>(
        &mut self,
        inferer: &'b dyn Inferer,
    ) -> anyhow::Result<HashMap<u64, Response<'b>>> {
        // pick up as many items as possible (by slicing the stores) and feed into the model.
        // this builds up a set of output stores that'll feed in sequence.
        let mut total_offset = 0;
        while self.scratch.batch_size > 0 {
            let preferred_batch_size = inferer.select_batch_size(self.scratch.batch_size);

            let mut view = self.scratch.chunk(total_offset, preferred_batch_size);

            inferer.infer_raw(&mut view)?;
            total_offset += preferred_batch_size;
        }

        let mut outputs = vec![Response::empty(); self.scratch.ids.len()];

        for slot in 0..inferer.output_shapes().len() {
            let slot_name = &inferer.output_shapes()[slot].0;

            assert_eq!(self.scratch.output_name(slot), slot_name);

            for (idx, o) in outputs.iter_mut().enumerate() {
                let slot_response = self.scratch.output_slot(slot, idx..idx + 1);
                o.data.insert(slot_name, slot_response.to_owned());
            }
        }

        Ok(self.scratch.ids.drain(..).zip(outputs).collect::<_>())
    }

    /// Check if there is any data to run on here.
    pub fn is_empty(&self) -> bool {
        self.scratch.batch_size == 0
    }

    /// Amount of elements to run on in the batch here.
    pub fn len(&self) -> usize {
        self.scratch.batch_size
    }
}
