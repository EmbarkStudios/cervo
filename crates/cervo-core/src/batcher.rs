// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 22 July 2022

/*!

*/

use std::{collections::HashMap, ops::Range};

use tract_core::tract_data::TVec;

use crate::inferer::{Inferer, Response, State};

struct ScratchPadData {
    name: String,
    data: Vec<f32>,
    count: usize,
}

impl ScratchPadData {
    fn new(name: String, count: usize, capacity: usize) -> Self {
        let mut this = Self {
            name,
            data: vec![],
            count,
        };

        this.reserve(capacity * count);
        this
    }

    fn reserve(&mut self, batch_size: usize) {
        self.data.resize(batch_size * self.count, 0.0);
    }

    #[inline]
    fn view(&self, range: Range<usize>) -> &[f32] {
        &self.data[range.start * self.count..range.end * self.count]
    }

    #[inline]
    fn view_mut(&mut self, range: Range<usize>) -> &mut [f32] {
        &mut self.data[range.start * self.count..range.end * self.count]
    }
}

pub struct ScratchPad {
    inputs: TVec<ScratchPadData>,
    outputs: TVec<ScratchPadData>,
    ids: Vec<u64>,
    batch_size: usize,
    capacity: usize,
}

const DEFAULT_CAPACITY: usize = 6;

impl ScratchPad {
    pub fn new_for_shapes(
        inputs: &[(String, Vec<usize>)],
        outputs: &[(String, Vec<usize>)],
    ) -> Self {
        Self::new_with_size(inputs, outputs, DEFAULT_CAPACITY)
    }

    pub fn new_with_size(
        inputs: &[(String, Vec<usize>)],
        outputs: &[(String, Vec<usize>)],
        capacity: usize,
    ) -> Self {
        let inputs = inputs
            .iter()
            .map(|(name, shape)| {
                let count = shape.iter().product();
                ScratchPadData::new(name.to_owned(), count, capacity)
            })
            .collect();

        let outputs = outputs
            .iter()
            .map(|(name, shape)| {
                let count = shape.iter().product();
                ScratchPadData::new(name.to_owned(), count, capacity)
            })
            .collect();

        Self {
            inputs,
            outputs,
            ids: vec![],
            batch_size: 0,
            capacity,
        }
    }

    pub fn next(&mut self, id: u64) {
        self.batch_size += 1;
        self.ids.push(id);

        if self.batch_size > self.capacity {
            self.capacity *= 2;

            for slot in &mut self.inputs {
                slot.reserve(self.capacity);
            }
        }
    }

    pub fn push(&mut self, slot: usize, data: Vec<f32>) {
        self.inputs[slot]
            .view_mut(self.batch_size - 1..self.batch_size)
            .copy_from_slice(&data);
    }

    pub fn chunk(&mut self, offset: usize, size: usize) -> ScratchPadView {
        let size = size.min(self.batch_size);
        self.batch_size -= size;

        ScratchPadView {
            pad: self,
            batch_range: offset..offset + size,
        }
    }

    #[inline]
    pub(crate) fn input_slot(&self, slot: usize, range: Range<usize>) -> &[f32] {
        self.inputs[slot].view(range)
    }

    #[inline]
    pub(crate) fn input_slot_mut(&mut self, slot: usize, range: Range<usize>) -> &mut [f32] {
        self.inputs[slot].view_mut(range)
    }

    #[inline]
    pub(crate) fn input_name(&self, slot: usize) -> &str {
        &self.inputs[slot].name
    }

    #[inline]
    pub(crate) fn output_slot(&self, slot: usize, range: Range<usize>) -> &[f32] {
        self.outputs[slot].view(range)
    }

    #[inline]
    pub(crate) fn output_slot_mut(&mut self, slot: usize, range: Range<usize>) -> &mut [f32] {
        self.outputs[slot].view_mut(range)
    }

    #[inline]
    pub(crate) fn output_name(&self, slot: usize) -> &str {
        &self.outputs[slot].name
    }
}

pub struct ScratchPadView<'a> {
    pad: &'a mut ScratchPad,
    batch_range: Range<usize>,
}

impl<'a> ScratchPadView<'a> {
    pub fn input_slot(&self, slot: usize) -> &[f32] {
        self.pad.input_slot(slot, self.batch_range.clone())
    }

    pub fn input_slot_mut(&mut self, slot: usize) -> &mut [f32] {
        self.pad.input_slot_mut(slot, self.batch_range.clone())
    }

    pub fn input_name(&self, slot: usize) -> &str {
        self.pad.input_name(slot)
    }

    pub fn output_slot(&self, slot: usize) -> &[f32] {
        self.pad.output_slot(slot, self.batch_range.clone())
    }

    pub fn output_slot_mut(&mut self, slot: usize) -> &mut [f32] {
        self.pad.output_slot_mut(slot, self.batch_range.clone())
    }

    pub fn output_name(&self, slot: usize) -> &str {
        self.pad.output_name(slot)
    }

    pub fn len(&self) -> usize {
        self.batch_range.len()
    }
}

pub struct Batcher {
    scratch: ScratchPad,
}

impl Batcher {
    pub fn new(inferer: &dyn Inferer) -> Self {
        Self {
            scratch: ScratchPad::new_for_shapes(inferer.input_shapes(), inferer.output_shapes()),
        }
    }

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

    pub fn execute<'a, 'b>(
        &'a mut self,
        inferer: &'b mut dyn Inferer,
    ) -> anyhow::Result<HashMap<u64, Response<'b>>> {
        let mut total_offset = 0;

        // pick up as many items as possible (by slicing the stores) and feed into the model.
        // this builds up a set of output stores that'll feed in sequence.
        while self.scratch.batch_size > 0 {
            let preferred_batch_size = inferer.select_batch_size(self.scratch.batch_size);

            let view = self.scratch.chunk(total_offset, preferred_batch_size);

            inferer.infer_raw(view)?;
            total_offset += preferred_batch_size;
        }

        let mut outputs = vec![Response::empty(); self.scratch.ids.len()];

        let output_count = inferer.output_shapes().len();
        for slot in 0..output_count {
            let slot_name = &inferer.output_shapes()[slot].0;

            assert_eq!(self.scratch.output_name(slot), slot_name);

            let mut total_offset = 0;
            for o in &mut outputs {
                let slot_response = self
                    .scratch
                    .output_slot(slot, total_offset..total_offset + 1);

                o.data.insert(slot_name, slot_response.to_owned());
                total_offset += 1;
            }
        }

        Ok(HashMap::from_iter(
            self.scratch.ids.drain(..).zip(outputs.into_iter()),
        ))
    }
}
