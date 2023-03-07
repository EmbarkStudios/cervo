use anyhow::Result;
use cervo_core::prelude::Inferer;
use cervo_runtime::BrainId;
use cervo_runtime::Runtime;
use std::time::Duration;
use std::time::Instant;

use std::{
    io::Write,
};

/// Measures the speedup obtained by using threading for the runtime.
pub(crate) fn compare_threading() -> Result<()> {
    let brain_repetition_values = vec![100];
    // let brain_repetition_values = vec![1, 10, 20, 50, 100];
    let batch_sizes = vec![32];
    // let batch_sizes = vec![2, 4, 8, 16, 32, 64];
    let onnx_paths = vec![
        "../../brains/test.onnx",
        "../../brains/test-large.onnx",
        "../../brains/test-complex.onnx",
    ];
    // TODO: Luc: Test this
    // println!("Current number of threads is {}", rayon::current_num_threads());
    // rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    // println!("Current number of threads is now {}", rayon::current_num_threads());
    let mut tester = Tester::new(brain_repetition_values, batch_sizes, onnx_paths);
    tester.run();

    // Get averaged values from tester.metrics

    let mut file = std::fs::File::create("temp.csv")?;

    for (avg_run, avg_run_for) in tester
        .metrics
        .average_run_speedup_per_core
        .iter()
        .zip(tester.metrics.average_run_for_speedup_per_core.iter())
    {
        writeln!(file, "{},{}", avg_run, avg_run_for)?;
    }
    // writeln!(
    //     file,
    //     "{:?},{},{}",
    //     row.kind,
    //     row.step,
    //     row.time.as_secs_f64() * 1e6 / denom
    // )?;

    Ok(())

    // TODO: Luc: Gather the results and plot them.
}

#[derive(Default)]
struct TesterState {
    runtime: Runtime,
    batch_size: usize,
    brain_repetitions: usize,
    duration: Duration,
    brain_ids: Vec<BrainId>,
    thread_count: usize,
}

enum Threaded {
    Threaded,
    SingleThreaded,
}

#[derive(Default)]
struct TesterMetrics {
    // Needs to be averaged
    run_speedup_per_core: Vec<Vec<f64>>,
    pub average_run_speedup_per_core: Vec<f64>,
    run_for_speedup_per_core: Vec<Vec<f64>>,
    pub average_run_for_speedup_per_core: Vec<f64>,
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
        let runtime = Runtime::new();
        let state = TesterState {
            runtime,
            duration: Duration::from_millis(5),
            ..Default::default()
        };
        let metrics = TesterMetrics {
            run_speedup_per_core: vec![Vec::new(); rayon::current_num_threads()],
            run_for_speedup_per_core: vec![Vec::new(); rayon::current_num_threads()],
            average_run_for_speedup_per_core: vec![0.0; rayon::current_num_threads()],
            average_run_speedup_per_core: vec![0.0; rayon::current_num_threads()],
        };
        Self {
            brain_repetition_values,
            batch_sizes,
            onnx_paths,
            state,
            metrics,
        }
    }

    fn run(&mut self) -> Option<()> {
        let max_threads = rayon::current_num_threads();
        for i in 10..max_threads {
            match rayon::ThreadPoolBuilder::new().num_threads(i + 1).build() {
                Err(_e) => None,
                Ok(pool) => Some(pool),
            }?
            .install(|| {
                // Set rayon threads to i + 1
                println!("Thread count is now {}", rayon::current_num_threads());
                self.state.thread_count = i;
                self.run_one_shot_tests();
                self.run_for_tests();
            });
        }
        self.metrics.average_run_speedup_per_core = self
            .metrics
            .run_speedup_per_core
            .iter()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect();
        self.metrics.average_run_for_speedup_per_core = self
            .metrics
            .run_for_speedup_per_core
            .iter()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect();
        Some(())
    }

    fn run_one_shot_tests(&mut self) {
        for brain_repetitions in self.brain_repetition_values.clone() {
            for batch_size in self.batch_sizes.clone() {
                println!(
                    "Running oneshot for {} threads, {} repetitions, {} batch size",
                    self.state.thread_count, brain_repetitions, batch_size
                );
                self.state.brain_repetitions = brain_repetitions;
                self.state.batch_size = batch_size;
                println!("Running threaded");
                let threaded_duration = self.run_one_shot(Threaded::Threaded);
                println!("Running single threaded");
                let single_duration = self.run_one_shot(Threaded::SingleThreaded);
                let speedup = single_duration.as_secs_f64() / threaded_duration.as_secs_f64();
                println!("Speed up is {}", speedup);
                self.metrics.run_speedup_per_core[self.state.thread_count].push(speedup);
            }
        }
    }

    fn run_for_tests(&mut self) {
        for brain_repetitions in self.brain_repetition_values.clone() {
            for batch_size in self.batch_sizes.clone() {
                println!(
                    "Running dur for {} threads, {} repetitions, {} batch size",
                    self.state.thread_count, brain_repetitions, batch_size
                );
                self.state.brain_repetitions = brain_repetitions;
                self.state.batch_size = batch_size;
                println!("Running threaded");
                let threaded_count = self.run_for(Threaded::Threaded);
                println!("Running single threaded");
                let unthreaded_count = self.run_for(Threaded::SingleThreaded);
                let speedup = threaded_count as f64 / unthreaded_count as f64;
                println!("Speed up is {}", speedup);
                self.metrics.run_for_speedup_per_core[self.state.thread_count].push(speedup);
            }
        }
    }

    fn run_for(&mut self, threaded: Threaded) -> usize {
        self.state.runtime = Runtime::new();
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
        self.state.runtime = Runtime::new();
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
                for (key, val) in observations.iter() {
                    self.state
                        .runtime
                        .push(*brain_id, *key, val.clone())
                        .expect(&format!(
                            "Could not push to runtime key: {}, val: {:?}",
                            key, val
                        ));
                }
            }
        }
    }
}
