// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 27 July 2022

use std::ops::Range;
use tract_core::tract_data::TVec;

/// Data container for a single slot in the scratchpad.
pub(super) struct ScratchPadData {
    /// The slot name in the model input
    pub(super) name: String,

    /// The data store
    pub(super) data: Vec<f32>,

    /// Number of data elements per batch-element.
    pub(super) count: usize,
}

impl ScratchPadData {
    /// Construct a new slot data with the specified capacity and element count.
    fn new(name: String, count: usize, capacity: usize) -> Self {
        let mut this = Self {
            name,
            data: vec![],
            count,
        };

        this.reserve(capacity);
        this
    }

    /// Reserve space for this many batch elemeents.
    fn reserve(&mut self, batch_size: usize) {
        self.data.resize(batch_size * self.count, 0.0);
    }

    /// A view over the specified range of batch elements.
    #[inline]
    fn view(&self, range: Range<usize>) -> &[f32] {
        &self.data[range.start * self.count..range.end * self.count]
    }

    /// A mutable view over the specified range of batch elements.
    #[inline]
    fn view_mut(&mut self, range: Range<usize>) -> &mut [f32] {
        &mut self.data[range.start * self.count..range.end * self.count]
    }
}

const DEFAULT_CAPACITY: usize = 6;
/// A scratch pad used during each inference call to avoid fragmented
/// allocations and copying.
pub struct ScratchPad {
    pub(super) inputs: TVec<ScratchPadData>,
    pub(super) outputs: TVec<ScratchPadData>,
    pub(super) ids: Vec<u64>,
    pub(super) batch_size: usize,
    capacity: usize,
}

impl ScratchPad {
    // TODO[TSolberg]: When switching to raw ModelAPI, fix this.
    /// Construct a new scratchpad for the provided API.
    pub fn new_for_shapes(
        inputs: &[(String, Vec<usize>)],
        outputs: &[(String, Vec<usize>)],
    ) -> Self {
        Self::new_with_size(inputs, outputs, DEFAULT_CAPACITY)
    }

    // TODO[TSolberg]: When switching to raw ModelAPI, fix this.
    /// Construct a new scratchpad for the provided API with a specified default capacity.
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

    /// Prepare the next slot to store data for the provided id.
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

    /// Push data for the specific slot.
    pub fn push(&mut self, slot: usize, data: Vec<f32>) {
        self.inputs[slot]
            .view_mut(self.batch_size - 1..self.batch_size)
            .copy_from_slice(&data);
    }

    /// View the chunk starting at batch-element `offset` containing `size` elements.x
    pub fn chunk(&mut self, offset: usize, size: usize) -> ScratchPadView<'_> {
        let size = size.min(self.batch_size);
        self.batch_size -= size;

        ScratchPadView {
            pad: self,
            batch_range: offset..offset + size,
        }
    }

    /// View of the specified `range` of input at location `slot`.
    #[inline]
    pub(crate) fn input_slot(&self, slot: usize, range: Range<usize>) -> &[f32] {
        self.inputs[slot].view(range)
    }

    /// A mutable view of the specified `range` of input at location `slot`.
    #[inline]
    pub(crate) fn input_slot_mut(&mut self, slot: usize, range: Range<usize>) -> &mut [f32] {
        self.inputs[slot].view_mut(range)
    }

    /// Retrieve the input name for `slot`.
    #[inline]
    pub(crate) fn input_name(&self, slot: usize) -> &str {
        &self.inputs[slot].name
    }

    /// View of the specified `range` of output at location `slot`.
    #[inline]
    pub(crate) fn output_slot(&self, slot: usize, range: Range<usize>) -> &[f32] {
        self.outputs[slot].view(range)
    }

    /// A mutable view of the specified `range` of output at location `slot`.
    #[inline]
    pub(crate) fn output_slot_mut(&mut self, slot: usize, range: Range<usize>) -> &mut [f32] {
        self.outputs[slot].view_mut(range)
    }

    /// Retrieve the output name for `slot`.
    #[inline]
    pub(crate) fn output_name(&self, slot: usize) -> &str {
        &self.outputs[slot].name
    }
}

/// A view over a set of batch elements in a scratch pad.
pub struct ScratchPadView<'a> {
    pad: &'a mut ScratchPad,
    batch_range: Range<usize>,
}

impl<'a> ScratchPadView<'a> {
    /// View of the input at location `slot`.
    pub fn input_slot(&self, slot: usize) -> &[f32] {
        self.pad.input_slot(slot, self.batch_range.clone())
    }

    /// A mutable view of the input at location `slot`.
    pub fn input_slot_mut(&mut self, slot: usize) -> &mut [f32] {
        self.pad.input_slot_mut(slot, self.batch_range.clone())
    }

    /// Retrieve the input name for `slot`.
    pub fn input_name(&self, slot: usize) -> &str {
        self.pad.input_name(slot)
    }

    /// A mutable view of the data at input `slot`.
    pub fn output_slot(&self, slot: usize) -> &[f32] {
        self.pad.output_slot(slot, self.batch_range.clone())
    }

    /// A mutable view of the data at location `slot`.
    pub fn output_slot_mut(&mut self, slot: usize) -> &mut [f32] {
        self.pad.output_slot_mut(slot, self.batch_range.clone())
    }

    /// Retrieve the output name for `slot`.
    pub fn output_name(&self, slot: usize) -> &str {
        self.pad.output_name(slot)
    }

    /// The batch size of this view.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.batch_range.len()
    }
}

#[cfg(test)]
mod tests {
    mod scratchppaddata {
        use super::super::ScratchPadData;
        #[test]
        fn has_right_initial_space() {
            let spd = ScratchPadData::new("epsilon".to_owned(), 24, 2);

            assert_eq!(spd.count, 24);
            assert_eq!(spd.data.len(), 48);
            assert_eq!(spd.name, "epsilon");
        }

        #[test]
        fn reserves_correct_size() {
            let mut spd = ScratchPadData::new("epsilon".to_owned(), 24, 2);

            spd.reserve(4);
            assert_eq!(spd.count, 24);
            assert_eq!(spd.data.len(), 24 * 4);
        }

        #[test]
        fn views_correct_range() {
            let mut spd = ScratchPadData::new("epsilon".to_owned(), 6, 4);

            spd.reserve(4);
            for idx in 0..24 {
                spd.data[idx] = idx as f32;
            }

            assert_eq!(spd.view(0..1), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(spd.view_mut(0..1), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(spd.view(1..2), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
            assert_eq!(spd.view_mut(1..2), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);

            assert_eq!(spd.view(3..4), [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
            assert_eq!(spd.view_mut(3..4), [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
        }
    }
}
