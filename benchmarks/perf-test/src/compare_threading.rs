use anyhow::Result;
use cervo_core::prelude::Inferer;
use cervo_runtime::BrainId;
use cervo_runtime::Runtime;
use std::time::Duration;
use std::time::Instant;
use std::fs::OpenOptions;

use std::{
    io::Write,
};

/// Measures the speedup obtained by using threading for the runtime.
pub(crate) fn compare_threading() -> Result<()> {
    let max_threads = rayon::current_num_threads();
    for i in 0..max_threads {
        // let i = 0;
        rayon::ThreadPoolBuilder::new().num_threads(i + 1).build()?
        .install(|| {
            let brain_repetition_values = vec![60];
            let batch_sizes = vec![32];
            let onnx_paths = vec![
                "../../brains/test.onnx",
                // "../../brains/test-large.onnx",
                // "../../brains/test-complex.onnx",
            ];
            // Create file if it doesn't exist yet: 
            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open("temp.csv")
                .unwrap();

            println!("Thread count is now {}", rayon::current_num_threads());
            let mut tester = Tester::new(brain_repetition_values, batch_sizes, onnx_paths);
            tester.run(i);

            let a = writeln!(file, "{},{},{}", i+1, tester.metrics.average_run_speedup, tester.metrics.average_run_for_speedup);
        });
    }

    // writeln!(file, "test2")?;



    Ok(())

    // TODO: Luc: Gather the results and plot them.
}

struct TesterState {
    runtime: Runtime,
    batch_size: usize,
    brain_repetitions: usize,
    duration: Duration,
    brain_ids: Vec<BrainId>,
}
impl Default for TesterState {
    fn default() -> Self {
        Self {
            runtime: Runtime::new(),
            batch_size: 32,
            brain_repetitions: 100,
            duration: Duration::from_millis(16),
            brain_ids: Vec::new(),
        }
    }
}

enum Threaded {
    Threaded,
    SingleThreaded,
}

#[derive(Default)]
struct TesterMetrics {
    // Needs to be averaged
    run_speedup: Vec<f64>,
    pub average_run_speedup: f64,
    run_for_speedup: Vec<f64>,
    pub average_run_for_speedup: f64,
}

#[derive(Default)]
struct Tester {
    brain_repetition_values: Vec<usize>,
    batch_sizes: Vec<usize>,
    onnx_paths: Vec<&'static str>,
    state: TesterState,
    pub metrics: TesterMetrics,
}

impl Tester {
    fn new(
        brain_repetition_values: Vec<usize>,
        batch_sizes: Vec<usize>,
        onnx_paths: Vec<&'static str>,
    ) -> Self {
        let state = TesterState::default();
        let metrics = TesterMetrics {
            run_speedup: Vec::new(),
            run_for_speedup: Vec::new(),
            average_run_for_speedup: 0.0,
            average_run_speedup: 0.0,
        };
        Self {
            brain_repetition_values,
            batch_sizes,
            onnx_paths,
            state,
            metrics,
        }
    }

    fn run(&mut self, thread_count: usize) -> Option<()> {
        self.state = TesterState::default();
        // Set rayon threads to i + 1
        self.run_one_shot_tests();
        self.run_for_tests();

        // Compute the average in metrics
        let mut run_speedup_sum = 0.0;
        let mut run_for_speedup_sum = 0.0;
        for i in 0..self.metrics.run_speedup.len() {
            run_speedup_sum += self.metrics.run_speedup[i];
            run_for_speedup_sum += self.metrics.run_for_speedup[i];
        }
        self.metrics.average_run_speedup = run_speedup_sum / self.metrics.run_speedup.len() as f64;
        self.metrics.average_run_for_speedup = run_for_speedup_sum / self.metrics.run_for_speedup.len() as f64;

        Some(())
    }

    fn run_one_shot_tests(&mut self) {
        for brain_repetitions in self.brain_repetition_values.clone() {
            for batch_size in self.batch_sizes.clone() {
                println!(
                    "Running oneshot for {} repetitions, {} batch size",
                    brain_repetitions, batch_size
                );
                self.state.brain_repetitions = brain_repetitions;
                self.state.batch_size = batch_size;
                println!("Running threaded");
                let threaded_duration = self.run_one_shot(Threaded::Threaded);
                println!("Running single threaded");
                let single_duration = self.run_one_shot(Threaded::SingleThreaded);
                let speedup = single_duration.as_secs_f64() / threaded_duration.as_secs_f64();
                println!("Speed up is {}", speedup);
                self.metrics.run_speedup.push(speedup);
            }
        }
    }

    fn run_for_tests(&mut self) {
        for brain_repetitions in self.brain_repetition_values.clone() {
            for batch_size in self.batch_sizes.clone() {
                println!(
                    "Running dur for {} repetitions, {} batch size",
                    brain_repetitions, batch_size
                );
                self.state.brain_repetitions = brain_repetitions;
                self.state.batch_size = batch_size;
                println!("Running threaded");
                let threaded_count = self.run_for(Threaded::Threaded);
                println!("Running single threaded");
                let unthreaded_count = self.run_for(Threaded::SingleThreaded);

                let speedup = threaded_count as f64 / unthreaded_count as f64;
                println!("Speed up is {} because {} / {}", speedup, threaded_count, unthreaded_count);
                self.metrics.run_for_speedup.push(speedup);
            }
        }
    }

    fn run_for(&mut self, threaded: Threaded) -> usize {
        self.state.runtime.clear();
        self.add_inferers_to_runtime();
        // Do a cold run
        self.state.runtime.run_threaded();

        self.push_tickets();

        let result = match threaded {
            Threaded::Threaded => self.state.runtime.run_for_threaded(self.state.duration),
            Threaded::SingleThreaded => self.state.runtime.run_for(self.state.duration),
        }
        .unwrap();
        result.len()
    }

    fn run_one_shot(&mut self, threaded: Threaded) -> Duration {
        self.state.runtime.clear();
        self.add_inferers_to_runtime();
        let start_time = Instant::now();
        match threaded {
            Threaded::Threaded => {
                let _ = self.state.runtime.run_threaded();
            }
            Threaded::SingleThreaded => {
                let _ = self.state.runtime.run();
            }
        };
        let elapsed_time = start_time.elapsed();
        elapsed_time
    }

    fn add_inferers_to_runtime(&mut self) {
        for _ in 0..self.state.brain_repetitions {
            for onnx_path in self.onnx_paths.iter() {
                let mut reader = crate::helpers::get_file(onnx_path).expect("Could not open file");
                let inferer = cervo_onnx::builder(&mut reader)
                    .build_fixed(&[self.state.batch_size])
                    .unwrap();

                let inputs = inferer.input_shapes().to_vec();
                let observations =
                    crate::helpers::build_inputs_from_desc(self.state.batch_size as u64, &inputs);
                let brain_id = self.state.runtime.add_inferer(inferer);

                for (key, val) in observations.iter() {
                    self.state
                        .runtime
                        .push(brain_id, *key, val.clone())
                        .expect(&format!(
                            "Could not push to runtime key: {}, val: {:?}",
                            key, val
                        ));
                    self.state.brain_ids.push(brain_id);
                }
            }
        }
    }

    /// Given an existing runtime with brains, push new tickets based on inputs.
    fn push_tickets(&mut self) {
        for brain_id in self.state.brain_ids.iter() {
            if let Some(input_shapes) = self.state.runtime.input_shapes(*brain_id).ok() {
                let input_shapes = input_shapes.clone().to_vec();
                let observations = crate::helpers::build_inputs_from_desc(
                    self.state.batch_size as u64,
                    &input_shapes,
                );
                for (key, val) in observations {
                    self.state
                        .runtime
                        .push(*brain_id, key, val)
                        .expect("Could not push to runtime");
                }
            }
        }
    }
}
