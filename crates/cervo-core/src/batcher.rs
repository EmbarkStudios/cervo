// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 22 July 2022

/*!

*/

use std::{collections::HashMap, ops::Range};

use tract_core::tract_data::TVec;

use crate::inferer::{BatchResponse, Inferer, Response, State};

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
    slots: TVec<ScratchPadData>,
    ids: Vec<u64>,
    batch_size: usize,
    capacity: usize,
}

const DEFAULT_CAPACITY: usize = 6;

impl ScratchPad {
    pub fn new_for_shapes(shapes: &[(String, Vec<usize>)]) -> Self {
        Self::new_with_size(shapes, DEFAULT_CAPACITY)
    }

    pub fn new_with_size(shapes: &[(String, Vec<usize>)], capacity: usize) -> Self {
        let slots = shapes
            .iter()
            .map(|(name, shape)| {
                let count = shape.iter().product();
                ScratchPadData::new(name.to_owned(), count, capacity)
            })
            .collect();

        Self {
            slots,
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

            for slot in &mut self.slots {
                slot.reserve(self.capacity);
            }
        }
    }

    pub fn push(&mut self, slot: usize, data: Vec<f32>) {
        self.slots[slot]
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
}

pub struct ScratchPadView<'a> {
    pad: &'a mut ScratchPad,
    batch_range: Range<usize>,
}

impl<'a> ScratchPadView<'a> {
    pub(crate) fn slot(&self, slot: usize) -> &[f32] {
        self.pad.slots[slot].view(self.batch_range.clone())
    }

    pub(crate) fn slot_mut(&mut self, slot: usize) -> &mut [f32] {
        self.pad.slots[slot].view_mut(self.batch_range.clone())
    }

    pub(crate) fn slot_name(&self, slot: usize) -> &str {
        &self.pad.slots[slot].name
    }

    pub(crate) fn len(&self) -> usize {
        self.batch_range.len()
    }
}

pub struct Batcher {
    scratch: ScratchPad,
    input_key_to_slot: Vec<String>,
    output_key_to_slot: Vec<String>,
}

impl Batcher {
    pub fn new(inferer: &dyn Inferer) -> Self {
        let input_key_to_slot: Vec<_> = inferer
            .input_shapes()
            .iter()
            .map(|(k, _)| k.clone())
            .collect();

        let output_key_to_slot: Vec<_> = inferer
            .output_shapes()
            .iter()
            .map(|(k, _)| k.clone())
            .collect();

        Self {
            scratch: ScratchPad::new_for_shapes(inferer.input_shapes()),
            input_key_to_slot,
            output_key_to_slot,
        }
    }

    pub fn new_sized(inferer: &dyn Inferer, size: usize) -> Self {
        let input_key_to_slot: Vec<_> = inferer
            .input_shapes()
            .iter()
            .map(|(k, _)| k.clone())
            .collect();

        let output_key_to_slot: Vec<_> = inferer
            .output_shapes()
            .iter()
            .map(|(k, _)| k.clone())
            .collect();

        Self {
            scratch: ScratchPad::new_with_size(inferer.input_shapes(), size),
            input_key_to_slot,
            output_key_to_slot,
        }
    }

    #[inline]
    fn slot(&self, name: &str) -> Option<usize> {
        self.input_key_to_slot.iter().position(|k| k == name)
    }

    /// Insert a single element into the batch to include in the next execution.
    pub fn push(&mut self, id: u64, state: State<'_>) -> anyhow::Result<()> {
        self.scratch.next(id);
        for (k, v) in state.data {
            let slot = self
                .slot(k)
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
        let mut response = BatchResponse::empty();

        for k in &self.output_key_to_slot {
            response.data.push((k, vec![]));
        }

        // pick up as many items as possible (by slicing the stores) and feed into the model.
        // this builds up a set of output stores that'll feed in sequence.
        while self.scratch.batch_size > 0 {
            let preferred_batch_size = inferer.select_batch_size(self.scratch.batch_size);

            let view = self.scratch.chunk(total_offset, preferred_batch_size);

            let batch_response = inferer.infer_raw(view)?;
            response.append(batch_response);
            total_offset += preferred_batch_size;
        }

        let mut output = HashMap::default();

        let mut total_offset = 0;
        for id in self.scratch.ids.drain(..) {
            let mut output_response = Response::empty();
            for ((idx, key), (name, shape)) in self
                .output_key_to_slot
                .iter()
                .enumerate()
                .zip(inferer.output_shapes())
            {
                assert_eq!(name, key);
                let batch_elements: usize = shape.iter().product();
                let slot_response = &response.data[idx];
                assert_eq!(slot_response.0, key);
                let batch_data = slot_response.1
                    [(total_offset * batch_elements)..(total_offset + 1) * batch_elements]
                    .to_owned();

                output_response.data.insert(name, batch_data);
            }

            total_offset += 1;
            output.insert(id, output_response);
        }

        Ok(output)
    }
}
