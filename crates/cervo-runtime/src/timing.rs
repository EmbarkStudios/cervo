// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 29 July 2022
use std::time::Duration;

pub(crate) struct TimingBucket {
    pub size: usize,
    state: WelfordState,
}

impl TimingBucket {
    pub(crate) fn new(size: usize, elapsed: Duration) -> TimingBucket {
        Self {
            size,
            state: WelfordState::new(elapsed),
        }
    }

    pub(crate) fn add(&mut self, elapsed: Duration) {
        self.state.update(elapsed);
    }

    pub(crate) fn mean(&self) -> Duration {
        self.state.mean()
    }

    pub(crate) fn scaled_mean(&self, to_size: usize) -> Duration {
        let ratio = self.size as f32 / to_size as f32;
        Duration::from_secs_f32(self.state.mean().as_secs_f32() / ratio)
    }
}

#[derive(Default)]
struct WelfordState {
    mean: f32,
    mean2: f32,

    count: usize,
}

impl WelfordState {
    fn new(elapsed: Duration) -> Self {
        let mut this = Self::default();
        this.update(elapsed);
        this
    }

    fn update(&mut self, value: Duration) {
        let value = value.as_secs_f32() * 1000.0;

        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / (self.count as f32);

        let delta2 = value - self.mean;
        self.mean2 += delta * delta2;
    }

    fn mean(&self) -> Duration {
        Duration::from_secs_f32(self.mean / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::WelfordState;
    use std::time::Duration;

    fn is_close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-5
    }

    #[test]
    fn initial_mean_initial_value() {
        let state = WelfordState::new(Duration::from_secs_f32(1.0));
        assert!(is_close(state.mean().as_secs_f32(), 1.0));
    }

    #[test]
    fn mean_no_diverge() {
        let mut state = WelfordState::new(Duration::from_secs_f32(1.0));

        for _ in 0..10 {
            state.update(Duration::from_secs_f32(1.0));
        }

        assert_eq!(state.mean().as_secs_f32(), 1.0);
    }

    #[test]
    fn mean_converge() {
        let mut state = WelfordState::new(Duration::from_secs_f32(0.0));

        for v in 1..10 {
            state.update(Duration::from_secs_f32(v as f32));
        }

        assert_eq!(state.mean().as_secs_f32(), 4.5);
    }

    #[test]
    fn mean_converge2() {
        let mut state = WelfordState::new(Duration::from_secs_f32(0.0));

        for v in 1..100 {
            state.update(Duration::from_secs_f32(v as f32));
        }

        assert_eq!(state.mean().as_secs_f32(), 49.5);
    }
}
