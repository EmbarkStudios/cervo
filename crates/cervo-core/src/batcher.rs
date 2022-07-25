// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 22 July 2022

/*!

*/

use std::collections::HashMap;

use crate::inferer::{Batch, BatchResponse, Inferer, Response, State};

pub struct Batcher {
    states: Vec<Option<Vec<f32>>>,
    ids: Vec<u64>,
    input_key_to_slot: Vec<String>,
    output_key_to_slot: Vec<String>,

    batch_sizes: usize,
}

impl Batcher {
    pub fn new_for_inferer(inferer: &dyn Inferer) -> Self {
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

        let states = vec![None; input_key_to_slot.len()];
        Self {
            states,
            ids: vec![],
            input_key_to_slot,
            output_key_to_slot,
            batch_sizes: 0,
        }
    }

    #[inline]
    fn slot(&self, name: &str) -> Option<usize> {
        self.input_key_to_slot.iter().position(|k| k == name)
    }

    /// Insert a single element into the batch to include in the next execution.
    pub fn push(&mut self, id: u64, state: State<'_>) -> anyhow::Result<()> {
        for (k, v) in state.data {
            let slot = self
                .slot(k)
                .ok_or_else(|| anyhow::anyhow!("key doesn't match an input: {:?}", k))?;

            if self.states[slot].is_none() {
                self.states[slot] = Some(vec![])
            }

            self.states[slot].as_mut().unwrap().extend_from_slice(&v);
        }

        self.ids.push(id);
        self.batch_sizes += 1;
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
        inferer: &'b mut impl Inferer,
    ) -> anyhow::Result<HashMap<u64, Response<'b>>> {
        let mut total_offset = 0;
        let mut response = BatchResponse::empty();

        for k in &self.output_key_to_slot {
            response.data.push((k, vec![]));
        }

        let empty = [];
        // pick up as many items as possible (by slicing the stores) and feed into the model.
        // this builds up a set of output stores that'll feed in sequence.
        while self.batch_sizes > 0 {
            let preferred_batch_size = inferer.select_batch_size(self.batch_sizes);
            self.batch_sizes -= preferred_batch_size;

            let mut batch = Batch::empty();
            batch.count = preferred_batch_size;

            for ((slot, (k, shape)), name) in inferer
                .input_shapes()
                .iter()
                .enumerate()
                .zip(self.input_key_to_slot.iter())
            {
                assert_eq!(k, name);
                let batch_elements: usize = shape.iter().product();

                // This is to deal with epsilon or other generative inferer wrappers.
                if let Some(store) = &self.states[slot] {
                    let batch_data = &store[(total_offset * batch_elements)
                        ..(total_offset + preferred_batch_size) * batch_elements];

                    batch.insert(name, batch_data);
                } else {
                    batch.insert(name, &empty);
                }
            }

            let batch_response = inferer.infer_batched(batch)?;
            response.append(batch_response);
            total_offset += preferred_batch_size;
        }

        self.states
            .iter_mut()
            .filter_map(|e| e.as_mut())
            .for_each(|e| e.clear());

        let mut output = HashMap::default();

        let mut total_offset = 0;
        for id in self.ids.drain(..) {
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
