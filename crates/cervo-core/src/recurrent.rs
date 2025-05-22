use std::collections::HashMap;

use crate::{batcher::ScratchPadView, inferer::Inferer};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use tract_core::tract_data::TVec;

pub struct RecurrentInfo {
    pub inkey: String,
    pub outkey: String,
}

struct RecurrentPair {
    inslot: usize,
    outslot: usize,
    numels: usize,
    offset: usize,
}

/// The [`RecurrentTracker`] wraps an inferer to manage states that
/// are input/output in a recurrent fashion, instead of roundtripping
/// them to the high-level code.
pub struct RecurrentTracker<T: Inferer> {
    inner: T,
    keys: TVec<RecurrentPair>,
    per_agent_states: RwLock<HashMap<u64, Box<[f32]>>>,
    agent_state_size: usize,
    // https://github.com/EmbarkStudios/cervo/issues/31
    inputs: Vec<(String, Vec<usize>)>,
    outputs: Vec<(String, Vec<usize>)>,
}

impl<T> RecurrentTracker<T>
where
    T: Inferer,
{
    /// Wraps the provided `inferer` to automatically track any keys that are both inputs/outputs.
    pub fn wrap(inferer: T) -> Result<RecurrentTracker<T>> {
        let inputs = inferer.input_shapes();
        let outputs = inferer.output_shapes();

        let mut keys = vec![];

        for (inkey, inshape) in inputs {
            for (outkey, outshape) in outputs {
                if inkey == outkey && inshape == outshape {
                    keys.push(RecurrentInfo {
                        inkey: inkey.clone(),
                        outkey: outkey.clone(),
                    })
                }
            }
        }

        Self::new(inferer, keys)
    }
}

impl<T> RecurrentTracker<T>
where
    T: Inferer,
{
    /// Create a new recurrency tracker for the model.
    ///
    pub fn new(inferer: T, info: Vec<RecurrentInfo>) -> Result<Self> {
        let inputs = inferer.input_shapes();
        let outputs = inferer.output_shapes();

        let mut offset = 0;
        let keys = info
            .iter()
            .map(|info| {
                let inslot = inputs
                    .iter()
                    .position(|input| info.inkey == input.0)
                    .with_context(|| format!("no input named {}", info.inkey))?;
                let outslot = outputs
                    .iter()
                    .position(|output| info.outkey == output.0)
                    .with_context(|| format!("no output named {}", info.outkey))?;

                let numels = inputs[inslot].1.iter().product();
                offset += numels;
                Ok(RecurrentPair {
                    inslot,
                    outslot,
                    numels,
                    offset: offset - numels,
                })
            })
            .collect::<Result<TVec<RecurrentPair>>>()?;

        let inputs = inputs
            .iter()
            .filter(|(k, _)| !info.iter().any(|info| &info.inkey == k))
            .cloned()
            .collect::<Vec<_>>();
        let outputs = outputs
            .iter()
            .filter(|(k, _)| !info.iter().any(|info| &info.outkey == k))
            .cloned()
            .collect::<Vec<_>>();
        Ok(Self {
            inner: inferer,
            keys,
            agent_state_size: offset,
            inputs,
            outputs,
            per_agent_states: Default::default(),
        })
    }
}

impl<T> Inferer for RecurrentTracker<T>
where
    T: Inferer,
{
    fn select_batch_size(&self, max_count: usize) -> usize {
        self.inner.select_batch_size(max_count)
    }

    fn infer_raw(&self, batch: &mut ScratchPadView<'_>) -> Result<(), anyhow::Error> {
        for pair in &self.keys {
            let (ids, indata) = batch.input_slot_mut_with_id(pair.inslot);

            let mut offset = 0;
            let states = self.per_agent_states.read();
            for id in ids {
                // if None, leave as zeros and pray
                if let Some(state) = states.get(id) {
                    indata[offset..offset + pair.numels]
                        .copy_from_slice(&state[pair.offset..pair.offset + pair.numels])
                }
                offset += pair.numels;
            }
        }

        self.inner.infer_raw(batch)?;

        for pair in &self.keys {
            let (ids, outdata) = batch.output_slot_mut_with_id(pair.outslot);

            let mut offset = 0;
            let mut states = self.per_agent_states.write();
            for id in ids {
                // if None, leave as zeros and pray
                if let Some(state) = states.get_mut(id) {
                    state[pair.offset..pair.offset + pair.numels]
                        .copy_from_slice(&outdata[offset..offset + pair.numels])
                }

                offset += pair.numels;
            }
        }

        Ok(())
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.inputs
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.outputs
    }

    fn begin_agent(&mut self, id: u64) {
        self.per_agent_states
            .write()
            .insert(id, vec![0.0; self.agent_state_size].into_boxed_slice());
        self.inner.begin_agent(id);
    }

    fn end_agent(&mut self, id: u64) {
        self.per_agent_states.write().remove(&id);
        self.inner.end_agent(id);
    }
}
