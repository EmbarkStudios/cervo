// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 22 September 2022

/*!

*/

mod ticket;

use crate::{error::CervoError, state::ModelState, AgentId, BrainId};
use ticket::Ticket;

use cervo_core::prelude::{Inferer, Response, State};
use std::{
    collections::{BinaryHeap, HashMap},
    time::{Duration, Instant},
};

/// The runtime wraps a multitude of inference models with batching support, and support for time-limited execution.
pub struct Runtime {
    models: Vec<ModelState>,
    queue: BinaryHeap<Ticket>,
    ticket_generation: u64,
    brain_generation: u16,
}

impl Runtime {
    /// Create a new empty runtime.
    pub fn new() -> Self {
        Self {
            models: Vec::with_capacity(16),
            queue: BinaryHeap::with_capacity(16),
            ticket_generation: 0,
            brain_generation: 0,
        }
    }

    /// Add a new inferer to this runtime.
    pub fn add_inferer(&mut self, inferer: impl Inferer + 'static + Send) -> BrainId {
        let id = BrainId(self.brain_generation);
        self.brain_generation += 1;

        self.models.push(ModelState::new(id, inferer));

        // New models always go to head of queue
        self.queue.push(Ticket(0, id));

        id
    }

    /// Queue the `state` to `brain` for `agent`, to be included in the next inference batch.
    pub fn push(
        &mut self,
        brain: BrainId,
        agent: AgentId,
        state: State<'_>,
    ) -> Result<(), CervoError> {
        match self.models.iter_mut().find(|m| m.id == brain) {
            Some(model) => model.push(agent, state),
            None => Err(CervoError::UnknownBrain(brain)),
        }
    }

    /// Run a single item through the specific `brain`. If there's
    /// pending data for the `brain`, this'll have some extra overhead
    /// for new allocations.
    pub fn infer_single(
        &mut self,
        brain_id: BrainId,
        state: State<'_>,
    ) -> Result<Response<'_>, CervoError> {
        match self.models.iter_mut().find(|m| m.id == brain_id) {
            Some(model) => model.infer_single(state),
            None => Err(CervoError::UnknownBrain(brain_id)),
        }
    }

    /// Executes all models with queued data.
    pub fn run(&mut self) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        let mut result = HashMap::default();

        for model in self.models.iter_mut() {
            if !model.needs_to_execute() {
                continue;
            }

            result.insert(model.id, model.run()?);
        }

        Ok(result)
    }

    /// Executes all models with queued data. Will attempt to keep
    /// total time below the provided duration, but due to noise or lack
    /// of samples might miss the deadline. See the note in [the root](./index.html).
    pub fn run_for(
        &mut self,
        mut duration: Duration,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        let mut result = HashMap::default();

        let mut any_executed = false;
        let mut executed: Vec<BrainId> = vec![];
        let mut non_executed = vec![];

        for ticket in self.queue.drain() {
            let res = match self.models.iter().find(|m| m.id == ticket.1) {
                Some(model) => {
                    if !model.needs_to_execute() || any_executed && !model.can_run_in_time(duration)
                    {
                        Ok(None)
                    } else {
                        let start = Instant::now();
                        let r = model.run();

                        let elapsed = start.elapsed();

                        duration = duration.saturating_sub(elapsed);

                        any_executed = true;
                        r.map(Some)
                    }
                }

                None => return Err(CervoError::UnknownBrain(ticket.1)),
            }?;

            match res {
                Some(res) => {
                    result.insert(ticket.1, res);
                    executed.push(ticket.1)
                }
                None => {
                    non_executed.push(ticket);
                }
            }
        }

        self.queue.extend(non_executed);
        for id in executed {
            let gen = self.ticket_generation;
            self.ticket_generation += 1;
            self.queue.push(Ticket(gen, id));
        }

        Ok(result)
    }

    /// Retrieve the output shapes for the provided brain.
    pub fn output_shapes(&self, brain: BrainId) -> Result<&[(String, Vec<usize>)], CervoError> {
        match self.models.iter().find(|m| m.id == brain) {
            Some(model) => Ok(model.inferer.output_shapes()),
            None => Err(CervoError::UnknownBrain(brain)),
        }
    }

    /// Retrieve the input shapes for the provided brain.
    pub fn input_shapes(&self, brain: BrainId) -> Result<&[(String, Vec<usize>)], CervoError> {
        match self.models.iter().find(|m| m.id == brain) {
            Some(model) => Ok(model.inferer.input_shapes()),
            None => Err(CervoError::UnknownBrain(brain)),
        }
    }

    /// Clear all models and all related data. Will error (after
    /// clearing *all* data) if there was queued items that are now
    /// orphaned.
    pub fn clear(&mut self) -> Result<(), CervoError> {
        // N.b. we don't clear brain generation; to avoid generational issues.
        self.queue.clear();
        self.ticket_generation = 0;

        let mut has_data = vec![];
        for model in self.models.drain(..) {
            if model.needs_to_execute() {
                has_data.push(model.id);
            }
        }

        if !has_data.is_empty() {
            Err(CervoError::OrphanedData(has_data))
        } else {
            Ok(())
        }
    }

    /// Clear a model and related data. Will error (after clearing
    /// *all* data) if there was queued items that are now orphaned.
    pub fn remove_inferer(&mut self, brain: BrainId) -> Result<(), CervoError> {
        // TODO[TSolberg]: when BinaryHeap::retain is stabilized, use that here.
        let mut to_repush = vec![];
        while !self.queue.is_empty() {
            // Safety: ^ must contain 1 item
            let elem = self.queue.pop().unwrap();

            if elem.1 == brain {
                break;
            } else {
                to_repush.push(elem);
            }
        }

        self.queue.extend(to_repush);

        if let Some(index) = self.models.iter().position(|state| state.id == brain) {
            // Safety: ^ we just found the index.
            let state = self.models.remove(index);
            if state.needs_to_execute() {
                Err(CervoError::OrphanedData(vec![brain]))
            } else {
                Ok(())
            }
        } else {
            Err(CervoError::UnknownBrain(brain))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::BrainId;

    use super::Runtime;
    use cervo_core::prelude::{Inferer, State};
    struct DummyInferer {
        sleep_duration: Duration,
    }

    impl Inferer for DummyInferer {
        fn select_batch_size(&self, count: usize) -> usize {
            assert_eq!(count, 1);
            count
        }

        fn infer_raw(
            &self,
            _batch: cervo_core::batcher::ScratchPadView<'_>,
        ) -> anyhow::Result<(), anyhow::Error> {
            std::thread::sleep(self.sleep_duration);
            Ok(())
        }

        fn input_shapes(&self) -> &[(String, Vec<usize>)] {
            &[]
        }

        fn output_shapes(&self) -> &[(String, Vec<usize>)] {
            &[]
        }
    }

    #[test]
    fn test_run_for_rotation() {
        let mut runtime = Runtime::new();
        let mut keys = vec![];
        for sleep in [0.02, 0.04, 0.06] {
            keys.push(runtime.add_inferer(DummyInferer {
                sleep_duration: Duration::from_secs_f32(sleep),
            }));
        }

        let push = |runtime: &mut Runtime, keys: &[BrainId]| {
            for k in keys {
                runtime.push(*k, 0, State::empty()).unwrap();
            }
        };

        for _ in 0..10 {
            push(&mut runtime, &keys);
            runtime.run().unwrap();
        }

        push(&mut runtime, &keys);
        let res = runtime.run_for(Duration::from_secs_f32(0.07)).unwrap();
        assert_eq!(res.len(), 2, "got keys: {:?}", res.keys());
        assert!(res.contains_key(&keys[0]));
        assert!(res.contains_key(&keys[1]));

        // queue should be 2, 0, 1
        let res = runtime.run_for(Duration::from_secs_f32(0.07)).unwrap();
        assert_eq!(res.len(), 1);
        assert!(res.contains_key(&keys[2]));

        push(&mut runtime, &keys);
        let res = runtime.run_for(Duration::from_secs_f32(0.07)).unwrap();
        assert_eq!(res.len(), 2, "got keys: {:?}", res.keys());
        assert!(res.contains_key(&keys[0]));
        assert!(res.contains_key(&keys[1]));
    }

    #[test]
    fn test_run_for_greedy() {
        let mut runtime = Runtime::new();
        let mut keys = vec![];
        for sleep in [0.02, 0.04, 0.06] {
            keys.push(runtime.add_inferer(DummyInferer {
                sleep_duration: Duration::from_secs_f32(sleep),
            }));
        }

        let push = |runtime: &mut Runtime, keys: &[BrainId]| {
            for k in keys {
                runtime.push(*k, 0, State::empty()).unwrap();
            }
        };

        for _ in 0..10 {
            push(&mut runtime, &keys);
            runtime.run().unwrap();
        }

        push(&mut runtime, &keys);
        let res = runtime.run_for(Duration::from_secs_f32(0.00)).unwrap();
        assert_eq!(res.len(), 1, "got keys: {:?}", res.keys());
        assert!(res.contains_key(&keys[0]));

        // queue should be 1, 2, 0
        let res = runtime.run_for(Duration::from_secs_f32(0.0)).unwrap();
        assert_eq!(res.len(), 1);
        assert!(res.contains_key(&keys[1]));

        // queue should be 2, 1, 0
        let res = runtime.run_for(Duration::from_secs_f32(0.0)).unwrap();
        assert_eq!(res.len(), 1);
        assert!(res.contains_key(&keys[2]));
    }
}
