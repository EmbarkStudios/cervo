use anyhow::Result;
use cervo_core::prelude::Inferer;
use cervo_runtime::BrainId;
use cervo_runtime::Runtime;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use std::io::Write;
// TODO: Luc: Average it
// Test: Is the speedup consistent with batch size change?

// statistical significance comes from _ repetitions

// Add 5 warmup runs where no data is recorded

// Show grid in the plot (pp.grid)

const RUN_COUNT: usize = 100;

type BatchSize = usize;

/// Measures the speedup obtained by using threading for the runtime.
pub(crate) fn compare_threading() -> Result<()> {
    let max_threads = rayon::current_num_threads();
    for i in 10..max_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(i + 1)
            .build()?
            .install(|| {
                println!("Thread count is now {}", rayon::current_num_threads());

                // let brain_repetition_values = vec![2];
                let brain_repetition_values = vec![1, 2, 5, 10];
                // let batch_sizes = vec![2];
                let batch_sizes = vec![1, 2, 3, 6, 8, 12, 16, 18];
                // TODO: Luc: Duration should be a parameter.
                let onnx_paths = vec![
                    "../../brains/test.onnx",
                    "../../brains/test-large.onnx",
                    "../../brains/test-complex.onnx",
                ];

                let mut tester =
                    Tester::new(brain_repetition_values, batch_sizes, onnx_paths, i + 1);
                tester.run(i);
            });
    }

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
            duration: Duration::from_millis(33), //TODO: Luc: Prepare duration
            brain_ids: Vec::new(),
        }
    }
}

enum Threaded {
    Threaded,
    SingleThreaded,
}

#[derive(Copy, Clone, Debug)]
enum Mode {
    OneShot,
    For,
}

#[derive(Default)]
struct TesterMetrics {
    // Needs to be averaged
    pub average_run_speedup: HashMap<BatchSize, f64>,
    pub average_run_for_speedup: HashMap<BatchSize, f64>,
}

#[derive(Default)]
struct Tester {
    brain_repetition_values: Vec<usize>,
    batch_sizes: Vec<usize>,
    onnx_paths: Vec<&'static str>,
    state: TesterState,
    num_cores: usize,
    pub metrics: TesterMetrics,
}

impl Tester {
    fn new(
        brain_repetition_values: Vec<usize>,
        batch_sizes: Vec<usize>,
        onnx_paths: Vec<&'static str>,
        num_cores: usize,
    ) -> Self {
        let state = TesterState::default();
        Self {
            brain_repetition_values,
            batch_sizes,
            onnx_paths,
            state,
            num_cores,
            metrics: Default::default(),
        }
    }

    fn run(&mut self, thread_count: usize) {
        // self.run_timed_tests(Mode::OneShot);
        self.run_timed_tests(Mode::For);
    }

    // TODO: Luc: Remove
    // fn run_one_shot_tests(&mut self) {
    //     for batch_size in self.batch_sizes.clone() {
    //         self.state.batch_size = batch_size;
    //         self.state.runtime.clear();
    //         self.add_inferers_to_runtime();

    //         let mut speedups = Vec::new();
    //         for i in 0..RUN_COUNT {
    //             self.push_tickets();
    //             let threaded_duration = self.run_one_shot(Threaded::Threaded);
    //             self.push_tickets();
    //             let single_duration = self.run_one_shot(Threaded::SingleThreaded);
    //             if i > 5 {
    //                 let speedup = single_duration.as_secs_f64() / threaded_duration.as_secs_f64();
    //                 speedups.push(speedup);
    //             }
    //         }

    //         // get average speedup
    //         let mut speedup_sum = 0.0;
    //         let speedup_len = speedups.len();
    //         for speedup in speedups {
    //             speedup_sum += speedup;
    //         }
    //         let speedup = speedup_sum / speedup_len as f64;
    //         println!(
    //             "Average Speed up for {} repetitions, {} batch size: {}",
    //             self.state.brain_repetitions, self.state.batch_size, speedup
    //         );

    //         self.metrics
    //             .average_run_speedup
    //             .entry(self.state.batch_size)
    //             .or_insert_with(|| speedup);
    //     }
    // }

    fn run_timed_tests(&mut self, mode: Mode) {
        self.state = TesterState::default();
        for brain_repetitions in self.brain_repetition_values.clone() {
            self.state.brain_repetitions = brain_repetitions;
            let name = match mode {
                Mode::OneShot => format!("run_tests_{}.csv", brain_repetitions),
                Mode::For => format!("run_for_tests_{}.csv", brain_repetitions),
            };
            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open(name)
                .unwrap();

            self.run_test(mode);

            let mut row = format!("{},", self.num_cores);
            for batch_size in self.batch_sizes.clone() {
                let val = match mode {
                    Mode::OneShot => self.metrics.average_run_speedup.get(&batch_size),
                    Mode::For => self.metrics.average_run_for_speedup.get(&batch_size),
                };

                if let Some(val) = val {
                    row += &format!("{},", val);
                }
            }

            let _ = writeln!(file, "{}", row);
        }
    }

    fn run_test(&mut self, mode: Mode) {
        for batch_size in self.batch_sizes.clone() {
            self.state.batch_size = batch_size;
            self.add_inferers_to_runtime();

            let mut speedups = Vec::new();

            for i in 0..RUN_COUNT {
                self.push_tickets();
                let threaded = match mode {
                    Mode::OneShot => self.run_one_shot(Threaded::Threaded).as_secs_f64(),
                    Mode::For => self.run_for(Threaded::Threaded) as f64,
                };
                self.push_tickets();
                let single = match mode {
                    Mode::OneShot => self.run_one_shot(Threaded::SingleThreaded).as_secs_f64(),
                    Mode::For => self.run_for(Threaded::SingleThreaded) as f64,
                };

                if i > 5 {
                    // This is because one shot measures time and for measures iterations
                    let speedup = match mode {
                        Mode::OneShot => single / threaded,
                        Mode::For => threaded / single,
                    };
                    speedups.push(speedup);
                }
            }

            // get average speedup
            let mut speedup_sum = 0.0;
            let speedup_len = speedups.len();
            for speedup in speedups {
                speedup_sum += speedup;
            }
            let speedup = speedup_sum / speedup_len as f64;
            println!(
                "For {} cores and Mode {:?}, Average Speed up for {} repetitions, {} batch size: {}",
                self.num_cores, mode, self.state.brain_repetitions, self.state.batch_size, speedup
            );

            match mode {
                Mode::OneShot => {
                    self.metrics
                        .average_run_speedup
                        .entry(self.state.batch_size)
                        .or_insert_with(|| speedup);

                },
                Mode::For => {
                    self.metrics
                        .average_run_for_speedup
                        .entry(self.state.batch_size)
                        .or_insert_with(|| speedup);
                },
            }
        }
    }

    fn run_for(&mut self, threaded: Threaded) -> usize {
        self.state.runtime.clear();
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
