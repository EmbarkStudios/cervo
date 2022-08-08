// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 28 July 2022

pub mod error;
mod state;
mod timing;

pub use crate::error::CervoError;
use crate::state::ModelState;
use cervo_core::prelude::{Inferer, Response, State};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    time::{Duration, Instant},
};

/// Identifier for a specific brain.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct BrainId(u16);

impl BrainId {
    pub unsafe fn into_inner(self) -> u16 {
        self.0
    }

    pub unsafe fn from_inner(value: u16) -> Self {
        Self(value)
    }
}

/// Identifier for a specific agent.
pub type AgentId = u64;

struct Ticket(u64, BrainId);

impl PartialEq for Ticket {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Ticket {}

impl PartialOrd for Ticket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for Ticket {
    fn cmp(&self, other: &Ticket) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Default)]
pub struct Runtime {
    models: Vec<ModelState>,
    queue: BinaryHeap<Ticket>,
    ticket_generation: u64,
    brain_generation: u16,
}

impl Runtime {
    pub fn push(&mut self, brain: BrainId, agent: AgentId, state: State) -> Result<(), CervoError> {
        match self.models.iter_mut().find(|m| m.id == brain) {
            Some(model) => {
                model.push(agent, state);
                Ok(())
            }
            None => Err(CervoError::UnknownBrain(brain)),
        }
    }

    pub fn add_inferer(&mut self, inferer: impl Inferer + 'static) -> BrainId {
        let id = BrainId(self.brain_generation);
        self.brain_generation += 1;

        self.models.push(ModelState::new(id, inferer));

        // New models always go to head of queue
        self.queue.push(Ticket(0, id));

        id
    }

    pub fn infer_single(
        &mut self,
        brain_id: BrainId,
        state: State,
    ) -> Result<Response, CervoError> {
        match self.models.iter_mut().find(|m| m.id == brain_id) {
            Some(model) => model.infer_single(state),
            None => Err(CervoError::UnknownBrain(brain_id)),
        }
    }

    pub fn run(&mut self) -> Result<HashMap<BrainId, HashMap<AgentId, Response>>, CervoError> {
        let mut result = HashMap::default();

        for model in self.models.iter_mut() {
            result.insert(model.id, model.run()?);
        }

        Ok(result)
    }

    pub fn run_for(
        &mut self,
        mut duration: Duration,
    ) -> Result<HashMap<BrainId, HashMap<AgentId, Response>>, CervoError> {
        let mut result = HashMap::default();

        let mut any_executed = false;
        let mut executed: Vec<BrainId> = vec![];
        let mut non_executed = vec![];

        for ticket in self.queue.drain() {
            // Break the lifetimes. :-) This is safe *assuming* that each model only has one ticket.

            let res = match self.models.iter().find(|m| m.id == ticket.1) {
                Some(model) => {
                    if !model.needs_to_execute() || any_executed && !model.can_run_in_time(duration)
                    {
                        Ok(None)
                    } else {
                        let start = Instant::now();
                        let r = model.run();

                        duration -= start.elapsed();
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
}
