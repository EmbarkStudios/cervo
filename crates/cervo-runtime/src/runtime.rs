// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 22 September 2022

/*!

*/

mod ticket;

use crate::{error::CervoError, state::ModelState, AgentId, BrainId};
use ticket::Ticket;

use cervo_core::prelude::{Inferer, Response, State};
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use std::{
    collections::{BinaryHeap, HashMap},
    time::{Duration, Instant},
};

/// The runtime wraps a multitude of inference models with batching support, and support for time-limited execution.
pub struct Runtime {
    pub models: Vec<ModelState>,
    queue: BinaryHeap<Ticket>,
    ticket_generation: u64,
    brain_generation: u16,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
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

    // pub fn brain_ids(&self) -> Vec<BrainId> {
    //     self.models.iter().map(|model| model.id).collect()
    // }

    // pub fn all_input_shapes(&self) -> Vec<Vec<(String, Vec<usize>)>> {
    //     self.models
    //         .iter()
    //         .map(|model| model.inferer.input_shapes().to_vec())
    //         .collect()
    // }

    /// Add a new inferer to this runtime. The new infererer will be at the end of the inference queue when using timed inference.
    pub fn add_inferer(&mut self, inferer: impl Inferer + 'static + Send) -> BrainId {
        let id = BrainId(self.brain_generation);
        self.brain_generation += 1;

        self.models.push(ModelState::new(id, inferer));

        // New models always go to head of queue
        self.queue.push(Ticket(self.ticket_generation, id));
        self.ticket_generation += 1;

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
        #[cfg(feature = "threaded")]
        {
            Ok(self.run_threaded())
        }
        #[cfg(not(feature = "threaded"))]
        {
            self.run_inner()
        }
    }

    /// Executes as many models as possible within the given duration.
    pub fn run_for(
        &mut self,
        duration: Duration,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        #[cfg(feature = "threaded")]
        {
            self.run_for_threaded(duration)
        }
        #[cfg(not(feature = "threaded"))]
        {
            self.run_for_inner(duration)
        }
    }

    pub fn run_threaded(&mut self) -> HashMap<BrainId, HashMap<AgentId, Response<'_>>> {
        // Use the iterator method from rayon
        self.models
            .par_iter_mut()
            .filter(|model| model.needs_to_execute())
            .map(|model| (model.id, model.run().unwrap()))
            .collect::<HashMap<BrainId, HashMap<AgentId, Response<'_>>>>()
    }

    /// Executes all models with queued data.
    pub fn run_non_threaded(
        &mut self,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        let mut result = HashMap::default();

        for model in self.models.iter_mut() {
            if !model.needs_to_execute() {
                continue;
            }

            result.insert(model.id, model.run()?);
        }

        Ok(result)
    }

    pub fn run_for_threaded(
        &mut self,
        duration: Duration,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        let start = Instant::now();
        let mut any_executed = false;

        let mut sorted_queue: Vec<Ticket> = Vec::with_capacity(self.queue.len());
        while !self.queue.is_empty() {
            sorted_queue.push(self.queue.pop().unwrap());
        }

        let queue = sorted_queue
            .iter()
            .filter_map(|ticket| {
                if let Some(model) = self.models.iter().find(|m| m.id == ticket.1) {
                    if model.needs_to_execute() || !any_executed {
                        any_executed = true;
                        return Some((ticket, model));
                    }
                }
                None
            })
            .collect::<Vec<(&Ticket, &ModelState)>>();

        let results = queue
            .into_par_iter()
            .map(|(ticket, model)| {
                if start.elapsed() > duration {
                    return None;
                }
                let time_remaining = duration.clone().saturating_sub(start.elapsed());
                if model.can_run_in_time(time_remaining) {
                    if let Some(r) = model.run().ok() {
                        return Some((ticket.1, r));
                    }
                }
                None
            })
            .flatten()
            .collect::<HashMap<BrainId, HashMap<AgentId, Response<'_>>>>();

        let finished = sorted_queue
            .iter()
            .filter(|ticket| results.contains_key(&ticket.1))
            .map(|ticket| {
                let gen = self.ticket_generation;
                self.ticket_generation += 1;
                Ticket(gen, ticket.1)
            })
            .collect::<Vec<Ticket>>();

        self.queue.clear();
        for ticket in sorted_queue {
            self.queue.push(ticket);
        }
        for ticket in finished {
            self.queue.push(ticket)
        }

        Ok(results)
    }

    /// Executes all models with queued data. Will attempt to keep
    /// total time below the provided duration, but due to noise or lack
    /// of samples might miss the deadline. See the note in [the root](./index.html).
    pub fn run_for_non_threaded(
        &mut self,
        mut duration: Duration,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response<'_>>>, CervoError> {
        let mut result = HashMap::default();

        let mut any_executed = false;
        let mut executed: Vec<BrainId> = vec![];
        let mut non_executed = vec![];

        while !self.queue.is_empty() {
            let ticket = self.queue.pop().unwrap();
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
                    executed.push(ticket.1);
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
    use super::Runtime;
    use crate::{BrainId, CervoError};
    use cervo_core::prelude::{Inferer, State};
    use std::time::Duration;
    use std::time::Instant;

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
        for sleep in [0.02, 0.04, 0.06, 0.04] {
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

        let res = runtime.run_for(Duration::from_secs_f32(0.07)).unwrap();
        assert_eq!(res.len(), 1);
        assert!(res.contains_key(&keys[2]));

        // queue should be 3, 0, 1, 2.
        // The below can run both 3 and 0 but only 3 has data.
        let res = runtime.run_for(Duration::from_secs_f32(0.07)).unwrap();
        assert_eq!(res.len(), 1);
        assert!(res.contains_key(&keys[3]));

        push(&mut runtime, &keys);
        let res = runtime.run_for(Duration::from_secs_f32(0.165)).unwrap();
        assert_eq!(res.len(), 4, "got keys: {:?}", res.keys());
        assert!(res.contains_key(&keys[0]));
        assert!(res.contains_key(&keys[1]));
        assert!(res.contains_key(&keys[2]));
        assert!(res.contains_key(&keys[3]));
    }

    #[test]
    fn test_run_skip_expensive() {
        let mut runtime = Runtime::new();
        let mut keys = vec![];
        for sleep in [0.02, 0.04, 0.06, 0.04] {
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
        let res = runtime.run_for(Duration::from_secs_f32(0.11)).unwrap();
        assert_eq!(res.len(), 3, "got keys: {:?}", res.keys());
        assert!(res.contains_key(&keys[0]));
        assert!(res.contains_key(&keys[1]));
        assert!(res.contains_key(&keys[3]));
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
        let res = runtime.run_for(Duration::from_secs_f32(0.0)).unwrap();
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

    #[test]
    fn test_run_single() {
        let mut runtime = Runtime::new();

        let k = runtime.add_inferer(DummyInferer {
            sleep_duration: Duration::from_secs_f32(0.01),
        });

        runtime.infer_single(k, State::empty()).unwrap();
        let r = runtime.run().unwrap();
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_run_single_with_push() {
        let mut runtime = Runtime::new();

        let k = runtime.add_inferer(DummyInferer {
            sleep_duration: Duration::from_secs_f32(0.01),
        });

        runtime.push(k, 0, State::empty()).unwrap();

        runtime.infer_single(k, State::empty()).unwrap();
        let mut r = runtime.run().unwrap();
        assert_eq!(r.len(), 1);
        let data = r.remove(&k).unwrap();

        assert_eq!(data.len(), 1);
        assert!(data.contains_key(&0));
    }

    #[test]
    fn unknown_brain_push() {
        let mut runtime = Runtime::new();
        let res = runtime.push(BrainId(10), 0, State::empty());

        assert!(res.is_err());
        let err = res.unwrap_err();

        if let CervoError::UnknownBrain(BrainId(10)) = err {
        } else {
            panic!("expected CervoError::UnknownBrain")
        }
    }

    #[test]
    fn unknown_brain_infer_single() {
        let mut runtime = Runtime::new();
        let res = runtime.infer_single(BrainId(10), State::empty());

        assert!(res.is_err());
        let err = res.unwrap_err();

        if let CervoError::UnknownBrain(BrainId(10)) = err {
        } else {
            panic!("expected CervoError::UnknownBrain")
        }
    }

    #[test]
    fn unknown_brain_remove() {
        let mut runtime = Runtime::new();
        let res = runtime.remove_inferer(BrainId(10));

        assert!(res.is_err());
        let err = res.unwrap_err();

        if let CervoError::UnknownBrain(BrainId(10)) = err {
        } else {
            panic!("expected CervoError::UnknownBrain")
        }
    }

    #[test]
    fn unknown_brain_remove_orphaned() {
        let mut runtime = Runtime::new();
        let k = runtime.add_inferer(DummyInferer {
            sleep_duration: Duration::from_secs_f32(0.1),
        });
        runtime.push(k, 0, State::empty()).unwrap();
        let res = runtime.remove_inferer(k);

        assert!(res.is_err());
        let err = res.unwrap_err();

        if let CervoError::OrphanedData(keys) = err {
            assert_eq!(keys, vec![k]);
        } else {
            panic!("expected CervoError::OrphanedData")
        }
    }

    #[test]
    fn unknown_brain_clear_orphaned() {
        let mut runtime = Runtime::new();
        let k = runtime.add_inferer(DummyInferer {
            sleep_duration: Duration::from_secs_f32(0.1),
        });
        runtime.push(k, 0, State::empty()).unwrap();
        let res = runtime.clear();

        assert!(res.is_err());
        let err = res.unwrap_err();

        if let CervoError::OrphanedData(keys) = err {
            assert_eq!(keys, vec![k]);
        } else {
            panic!("expected CervoError::OrphanedData")
        }
    }
}
