use std::collections::HashMap;

use crate::{batcher::ScratchPadView, inferer::Inferer};
use anyhow::{Context, Result};
use itertools::Itertools;
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
        let inputs = inferer.raw_input_shapes();
        let outputs = inferer.raw_output_shapes();

        let mut keys = vec![];

        for (inkey, inshape) in inputs {
            for (outkey, outshape) in outputs {
                if inkey == outkey && inshape == outshape {
                    keys.push(RecurrentInfo {
                        inkey: inkey.clone(),
                        outkey: outkey.clone(),
                    });
                }
            }
        }

        if keys.is_empty() {
            let inkeys = inputs.iter().map(|(k, _)| k).join(", ");
            let outkeys = outputs.iter().map(|(k, _)| k).join(", ");
            anyhow::bail!(
                "Unable to find a matching key between inputs [{inkeys}] and outputs [{outkeys}]"
            );
        }
        Self::new(inferer, keys)
    }

    /// Create a new recurrency tracker for the model.
    ///
    pub fn new(inferer: T, info: Vec<RecurrentInfo>) -> Result<Self> {
        let inputs = inferer.raw_input_shapes();
        let outputs = inferer.raw_output_shapes();

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
                        .copy_from_slice(&state[pair.offset..pair.offset + pair.numels]);
                } else {
                    indata[offset..offset + pair.numels].fill(0.0);
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
                        .copy_from_slice(&outdata[offset..offset + pair.numels]);
                }

                offset += pair.numels;
            }
        }

        Ok(())
    }

    fn raw_input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.raw_input_shapes()
    }

    fn raw_output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inner.raw_output_shapes()
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

#[cfg(test)]
mod tests {

    use crate::{
        batcher::ScratchPadView,
        inferer::State,
        prelude::{Batcher, Inferer},
    };

    use super::RecurrentTracker;

    struct DummyInferer {
        end_called: bool,
        begin_called: bool,
        inputs: Vec<(String, Vec<usize>)>,
        outputs: Vec<(String, Vec<usize>)>,
    }

    impl Default for DummyInferer {
        fn default() -> Self {
            Self::new_named(
                "lstm_hidden_state",
                "lstm_cell_state",
                "lstm_hidden_state",
                "lstm_cell_state",
            )
        }
    }

    impl DummyInferer {
        fn new_named(
            hidden_name_in: &str,
            cell_name_in: &str,
            hidden_name_out: &str,
            cell_name_out: &str,
        ) -> Self {
            Self {
                end_called: false,
                begin_called: false,
                inputs: vec![
                    (hidden_name_in.to_owned(), vec![2, 1]),
                    (cell_name_in.to_owned(), vec![2, 3]),
                ],
                outputs: vec![
                    (hidden_name_out.to_owned(), vec![2, 1]),
                    (cell_name_out.to_owned(), vec![2, 3]),
                    ("hidden_output".to_owned(), vec![2]),
                    ("cell_output".to_owned(), vec![6]),
                ],
            }
        }
    }

    impl Inferer for DummyInferer {
        fn select_batch_size(&self, _max_count: usize) -> usize {
            1
        }

        fn infer_raw(&self, batch: &mut ScratchPadView<'_>) -> anyhow::Result<(), anyhow::Error> {
            assert_eq!(batch.inner().input_name(0), "lstm_hidden_state");
            let hidden_value = batch.input_slot(0);
            let hidden_new = hidden_value.iter().map(|v| *v + 1.0).collect::<Vec<_>>();

            assert_eq!(batch.inner().output_name(0), "lstm_hidden_state");
            batch.output_slot_mut(0).copy_from_slice(&hidden_new);

            assert_eq!(batch.inner().input_name(1), "lstm_cell_state");
            let cell_value = batch.input_slot(1);
            let cell_new = cell_value.iter().map(|v| *v + 2.0).collect::<Vec<_>>();

            assert_eq!(batch.inner().output_name(1), "lstm_cell_state");
            batch.output_slot_mut(1).copy_from_slice(&cell_new);

            assert_eq!(batch.inner().output_name(2), "hidden_output");
            let hidden = batch.output_slot_mut(2);
            hidden.copy_from_slice(&hidden_new);

            assert_eq!(batch.inner().output_name(3), "cell_output");
            let cell = batch.output_slot_mut(3);
            cell.copy_from_slice(&cell_new);

            Ok(())
        }

        fn raw_input_shapes(&self) -> &[(String, Vec<usize>)] {
            &self.inputs
        }

        fn raw_output_shapes(&self) -> &[(String, Vec<usize>)] {
            &self.outputs
        }

        fn begin_agent(&mut self, _id: u64) {
            self.begin_called = true;
        }
        fn end_agent(&mut self, _id: u64) {
            self.end_called = true;
        }
    }

    #[test]
    fn begin_end_forwarded() {
        let inferer = DummyInferer::default();
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        assert!(recurrent.inner.begin_called);

        recurrent.end_agent(10);
        assert!(recurrent.inner.end_called);
    }

    #[test]
    fn begin_creates_state() {
        let inferer = DummyInferer::default();
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        assert!(recurrent.per_agent_states.read().contains_key(&10));
    }

    #[test]
    fn end_removes_state() {
        let inferer = DummyInferer::default();
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        recurrent.end_agent(10);

        assert!(!recurrent.per_agent_states.read().contains_key(&10));
    }

    #[test]
    fn wrap_warns_no_keys() {
        let inferer = DummyInferer::new_named("a", "b", "c", "d");
        let should_err = RecurrentTracker::wrap(inferer);
        assert!(should_err.is_err());
    }

    #[test]
    fn test_infer() {
        let inferer = DummyInferer::default();
        let mut batcher = Batcher::new(&inferer);
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        batcher.push(10, State::empty()).unwrap();

        batcher.execute(&recurrent).unwrap();
    }

    #[test]
    fn test_infer_output() {
        let inferer = DummyInferer::default();
        let mut batcher = Batcher::new(&inferer);
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        batcher.push(10, State::empty()).unwrap();

        let res = batcher.execute(&recurrent).unwrap();
        let agent_data = &res[&10];
        assert!(agent_data.data.contains_key("hidden_output"));
        assert!(agent_data.data.contains_key("cell_output"));

        assert!(agent_data.data["hidden_output"].iter().all(|v| *v == 1.0));
        assert!(agent_data.data["cell_output"].iter().all(|v| *v == 2.0));
    }

    #[test]
    fn test_infer_twice_output() {
        let inferer = DummyInferer::default();
        let mut batcher = Batcher::new(&inferer);
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        batcher.push(10, State::empty()).unwrap();

        batcher.execute(&recurrent).unwrap();
        batcher.push(10, State::empty()).unwrap();

        let res = batcher.execute(&recurrent).unwrap();

        let agent_data = &res[&10];
        assert!(agent_data.data.contains_key("hidden_output"));
        assert!(agent_data.data.contains_key("cell_output"));

        assert!(agent_data.data["hidden_output"].iter().all(|v| *v == 2.0));
        assert!(agent_data.data["cell_output"].iter().all(|v| *v == 4.0));
    }

    #[test]
    fn test_infer_twice_reuse_id() {
        let inferer = DummyInferer::default();
        let mut batcher = Batcher::new(&inferer);
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        batcher.push(10, State::empty()).unwrap();
        batcher.execute(&recurrent).unwrap();

        recurrent.end_agent(10);

        recurrent.begin_agent(10);

        batcher.push(10, State::empty()).unwrap();

        let res = batcher.execute(&recurrent).unwrap();
        let agent_data = &res[&10];

        assert!(agent_data.data.contains_key("hidden_output"));
        assert!(agent_data.data.contains_key("cell_output"));

        assert!(agent_data.data["hidden_output"].iter().all(|v| *v == 1.0));
        assert!(agent_data.data["cell_output"].iter().all(|v| *v == 2.0));
    }

    #[test]
    fn test_infer_multiple_agents() {
        let inferer = DummyInferer::default();
        let mut batcher = Batcher::new(&inferer);
        let mut recurrent = RecurrentTracker::wrap(inferer).unwrap();

        recurrent.begin_agent(10);
        recurrent.begin_agent(20);
        batcher.push(10, State::empty()).unwrap();
        batcher.push(20, State::empty()).unwrap();
        batcher.execute(&recurrent).unwrap();

        recurrent.begin_agent(20);
        batcher.push(10, State::empty()).unwrap();
        batcher.push(20, State::empty()).unwrap();
        batcher.execute(&recurrent).unwrap();

        recurrent.begin_agent(30);
        batcher.push(10, State::empty()).unwrap();
        batcher.push(30, State::empty()).unwrap();
        let res = batcher.execute(&recurrent).unwrap();
        let agent_data = &res[&10];

        assert!(agent_data.data.contains_key("hidden_output"));
        assert!(agent_data.data.contains_key("cell_output"));

        assert!(agent_data.data["hidden_output"].iter().all(|v| *v == 3.0));
        assert!(agent_data.data["cell_output"].iter().all(|v| *v == 6.0));

        let agent_data = &res[&30];

        assert!(agent_data.data.contains_key("hidden_output"));
        assert!(agent_data.data.contains_key("cell_output"));

        assert!(agent_data.data["hidden_output"].iter().all(|v| *v == 1.0));
        assert!(agent_data.data["cell_output"].iter().all(|v| *v == 2.0));
    }
}
