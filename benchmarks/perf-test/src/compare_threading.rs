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
// TODO: Luc:
// Why does a batch size of 16 seem optimal? Is it because of the observation size of the inferrers? Investigate.

const RUN_COUNT: usize = 30;

type BatchSize = usize;

/// Measures the speedup obtained by using threading for the runtime.
pub(crate) fn compare_threading() -> Result<()> {
    let max_threads = rayon::current_num_threads();
    for i in 0..max_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(i + 1)
            .build()?
            .install(|| {
                println!("Thread count is now {}", rayon::current_num_threads());

                // let brain_repetition_values = vec![5, 10];
                // let brain_repetition_values = vec![10];
                let brain_repetition_values = vec![5, 10, 15];
                // let brain_repetition_values = vec![1, 2, 5, 10];
                // let batch_sizes = vec![8];
                let batch_sizes = vec![1, 2, 3, 6, 8, 12, 16, 18];
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
            duration: Duration::from_millis(500),
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
                .append(self.num_cores > 1)
                .truncate(self.num_cores == 1)
                .create(true)
                .open(name)
                .unwrap();

            // Write header row with batch sizes
            if self.num_cores == 1 {
                let mut header = "num_cores,".to_string();
                for batch_size in self.batch_sizes.clone() {
                    header += &format!("b_{},", batch_size);
                }
                header.pop();
                let _ = writeln!(file, "{}", header);
            }

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
            row.pop();

            let _ = writeln!(file, "{}", row);
        }
    }

    fn set_duration(&mut self) {
        let previous_brain_repetitions = self.state.brain_repetitions;

        self.state.brain_repetitions = 1;
        self.state.runtime.clear();
        self.add_inferers_to_runtime();

        let mut durations = Vec::new();

        for i in 0..10 {
            self.push_tickets();
            let duration = self.run_one_shot(Threaded::SingleThreaded);
            durations.push(duration.as_nanos() as f64);
        }

        let average = Self::average(durations);
        self.state.duration = Duration::from_nanos(average as u64);
        self.state.brain_repetitions = previous_brain_repetitions;
        println!("Duration is now {:?}", self.state.duration);
    }

    fn average(values: Vec<f64>) -> f64 {
        let sum: f64 = values.iter().sum();
        sum / values.len() as f64
    }

    fn run_test(&mut self, mode: Mode) {
        for batch_size in self.batch_sizes.clone() {
            self.state.batch_size = batch_size;

            let mut speedups = Vec::new();
            let mut threaded_num = Vec::new();
            let mut single_num = Vec::new();

            // Determine the duration required to obtain a solid denominator
            if let Mode::For = mode {
                self.set_duration();
            }

            self.state.runtime.clear();
            self.add_inferers_to_runtime();

            for i in 0..RUN_COUNT {
                self.push_tickets();

                let single = match mode {
                    Mode::OneShot => self.run_one_shot(Threaded::SingleThreaded).as_secs_f64(),
                    Mode::For => {
                        let results = self.run_for(Threaded::SingleThreaded) as f64;
                        // Clear remaining tickets
                        self.run_one_shot(Threaded::Threaded);
                        results
                    }
                };
                self.push_tickets();
                let threaded = match mode {
                    Mode::OneShot => self.run_one_shot(Threaded::Threaded).as_secs_f64(),
                    Mode::For => {
                        let results = self.run_for(Threaded::Threaded) as f64;
                        // Clear remaining tickets
                        self.run_one_shot(Threaded::Threaded);
                        results
                    }
                };

                if i > 5 {
                    // This is because one shot measures time and for measures iterations
                    let speedup = match mode {
                        Mode::OneShot => single / threaded,
                        Mode::For => {
                            // println!("Threaded: {}, Single: {}", threaded, single);
                            threaded / single
                        }
                    };
                    threaded_num.push(threaded);
                    single_num.push(single);
                    speedups.push(speedup);
                }
            }

            let speedup = Self::average(speedups);
            let single_num = Self::average(single_num);
            let threaded_num = Self::average(threaded_num);
            println!(
                "For {} cores and Mode {:?}, Average Speed up for {} repetitions, {} batch size: {} (with single: {}, threaded: {})",
                self.num_cores, mode, self.state.brain_repetitions, self.state.batch_size, speedup, single_num, threaded_num
            );

            let _ = match mode {
                Mode::OneShot => self
                    .metrics
                    .average_run_speedup
                    .insert(self.state.batch_size, speedup),
                Mode::For => self
                    .metrics
                    .average_run_for_speedup
                    .insert(self.state.batch_size, speedup),
            };
        }
    }

    fn run_for(&mut self, threaded: Threaded) -> usize {
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
        for onnx_path in self.onnx_paths.iter() {
            let mut reader = crate::helpers::get_file(onnx_path).expect("Could not open file");

            for _ in 0..self.state.brain_repetitions {
                reader
                    .seek(std::io::SeekFrom::Start(0))
                    .expect("seeking to start of file");
                let inferer = cervo_onnx::builder(&mut reader)
                    .build_fixed(&[self.state.batch_size])
                    .unwrap();

                let brain_id = self.state.runtime.add_inferer(inferer);
                self.state.brain_ids.push(brain_id);
            }
        }
    }

    /// Given an existing runtime with brains, push new tickets based on inputs.
    fn push_tickets(&mut self) {
        let TesterState {
            runtime,
            brain_ids,
            batch_size,
            ..
        } = &mut self.state;
        for brain_id in brain_ids.iter() {
            if let Some(input_shapes) = runtime.input_shapes(*brain_id).ok() {
                let input_shapes = input_shapes.to_vec();
                let observations =
                    crate::helpers::build_inputs_from_desc(*batch_size as u64, &input_shapes);
                for (key, val) in observations {
                    runtime
                        .push(*brain_id, key, val)
                        .expect("Could not push to runtime");
                }
            }
        }
    }
}
