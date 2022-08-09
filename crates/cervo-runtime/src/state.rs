// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 29 July 2022

/*!

*/

use crate::{timing::TimingBucket, AgentId};
use cervo_core::prelude::{Batcher, Inferer, InfererExt, Response, State};
use std::{
    cell::RefCell,
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::{error::CervoError, BrainId};

pub(crate) struct ModelState {
    pub(crate) id: BrainId,
    inferer: Box<dyn Inferer + 'static>,
    batcher: RefCell<Batcher>,
    timings: RefCell<Vec<TimingBucket>>,
}

impl ModelState {
    pub(crate) fn new(id: BrainId, inferer: impl Inferer + 'static) -> Self {
        let batcher = RefCell::new(Batcher::new(&inferer));
        Self {
            id,
            inferer: Box::new(inferer),
            batcher,
            timings: RefCell::new(vec![]),
        }
    }

    pub(crate) fn push(&mut self, agent_id: AgentId, state: State<'_>) -> Result<(), CervoError> {
        let mut batcher = self.batcher.borrow_mut();
        batcher.push(agent_id, state).map_err(CervoError::Internal)
    }

    pub(crate) fn needs_to_execute(&self) -> bool {
        !self.batcher.borrow().is_empty()
    }

    pub(crate) fn can_run_in_time(&self, duration: Duration) -> bool {
        if self.timings.borrow().is_empty() {
            return true;
        }

        let size = self.batcher.borrow().len();
        let timings = self.timings.borrow();
        let partition = timings.partition_point(|b| b.size <= size);

        if partition == timings.len() {
            let last = timings.last().unwrap();
            last.scaled_mean(size) <= duration
        } else {
            let elem = &timings[partition];
            if elem.size == size {
                elem.mean() <= duration
            } else if partition == 0 {
                let elem = &timings[partition];
                elem.scaled_mean(size) <= duration
            } else {
                let elem = &timings[partition - 1];
                elem.scaled_mean(size) <= duration
            }
        }
    }

    pub(crate) fn infer_single<'a>(
        &'a mut self,
        state: State<'_>,
    ) -> Result<Response<'a>, CervoError> {
        let start = Instant::now();
        let mut batcher = self.batcher.borrow_mut();

        let res = if batcher.is_empty() {
            batcher.push(0, state).map_err(CervoError::Internal)?;

            let mut res = batcher
                .execute(&self.inferer)
                .map_err(CervoError::Internal)?;

            res.remove(&0).ok_or_else(|| {
                CervoError::Internal(anyhow::anyhow!(
                    "fatal error, no data when data was expected"
                ))
            })
        } else {
            self.inferer
                .infer_single(state)
                .map_err(CervoError::Internal)
        }?;

        let elapsed = start.elapsed();
        let mut timings = self.timings.borrow_mut();
        match timings.iter_mut().find(|b| b.size == 1) {
            Some(bucket) => bucket.add(elapsed),
            None => {
                timings.push(TimingBucket::new(1, elapsed));
                timings.sort_by_key(|b| b.size);
            }
        }

        Ok(res)
    }

    pub(crate) fn run(&self) -> Result<HashMap<AgentId, Response<'_>>, CervoError> {
        let mut batcher = self.batcher.borrow_mut();

        if batcher.is_empty() {
            return Ok(HashMap::default());
        }

        let start = Instant::now();
        let batch_size = batcher.len();

        let res = batcher
            .execute(&self.inferer)
            .map_err(CervoError::Internal)?;

        let elapsed = start.elapsed();
        let mut timings = self.timings.borrow_mut();
        match timings.iter_mut().find(|b| b.size == batch_size) {
            Some(bucket) => bucket.add(elapsed),
            None => {
                timings.push(TimingBucket::new(batch_size, elapsed));
                timings.sort_by_key(|b| b.size);
            }
        }

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use cervo_core::prelude::{Batcher, Inferer, State};

    use super::ModelState;
    use crate::{timing::TimingBucket, BrainId};

    struct DummyInferer;

    impl Inferer for DummyInferer {
        fn select_batch_size(&self, _max_count: usize) -> usize {
            0
        }

        fn infer_raw(
            &self,
            _batch: cervo_core::batcher::ScratchPadView<'_>,
        ) -> anyhow::Result<(), anyhow::Error> {
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
    fn can_fit_yes() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![TimingBucket::new(1, Duration::from_secs(1))].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs(1)));
    }

    #[test]
    fn can_fit_yes_extrapolate() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![TimingBucket::new(1, Duration::from_secs(1))].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs(2)));
    }

    #[test]
    fn can_fit_no_extrapolate() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![TimingBucket::new(1, Duration::from_secs(1))].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs(2)));
    }

    #[test]
    fn can_fit_empty() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs(2)));
    }

    #[test]
    fn can_fit_yes_after() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![TimingBucket::new(2, Duration::from_secs(1))].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs_f32(0.5)));
    }

    #[test]
    fn can_fit_no_after() {
        let batcher = Batcher::new(&DummyInferer).into();
        let state = ModelState {
            id: BrainId(0),
            inferer: Box::new(DummyInferer),
            batcher,
            timings: vec![TimingBucket::new(2, Duration::from_secs(1))].into(),
        };

        state.batcher.borrow_mut().push(0, State::empty()).unwrap();
        assert!(state.can_run_in_time(Duration::from_secs_f32(0.6)));
    }
}
