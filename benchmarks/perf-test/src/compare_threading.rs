use cervo_core::prelude::Inferer;
use cervo_core::prelude::Response;
use std::collections::HashMap;
use cervo_runtime::BrainId;
use cervo_runtime::AgentId;
use cervo_runtime::Runtime;
use std::time::Duration;
use std::time::Instant;

/// Given an existing runtime with brains, push new tickets based on inputs.
fn push_tickets(runtime: &mut Runtime, batch_size: usize) {
    let mut observation_vec = vec![];
    for (inputs, brain_id) in runtime.models.iter().map(|model| {
        (model.inferer.input_shapes().to_vec(), model.id)
    })
    {
        let observations = crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs);
        for (key, val) in observations.iter() {
            observation_vec.push((brain_id.clone(), &key, val.clone()));
        }
    }
    for (brain_id, key, val) in observation_vec {
        runtime
        // TODO: LUc: Entrypoint:
            .push(brain_id, *key.clone(), val)
            .expect(&format!(
                "Could not push to runtime key: {}, val: {:?}",
                key, val
            ));
    }
}


/// Create a new runtime with for `brain_repetitions` times the brains in `onnx_paths`.
/// Also generates observations based on the inferers' input shapes.
fn add_inferers_to_runtime(
    runtime: &mut Runtime,
    onnx_paths: &[&str],
    brain_repetitions: usize,
    batch_size: usize,
) {
    for _ in 0..brain_repetitions {
        for onnx_path in onnx_paths {
            let mut reader = crate::helpers::get_file(onnx_path).expect("Could not open file");
            let inferer = cervo_onnx::builder(&mut reader)
                .build_fixed(&[batch_size])
                .unwrap();

            let inputs = inferer.input_shapes().to_vec();
            let observations = crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs);
            let brain_id = runtime.add_inferer(inferer);

            for (key, val) in observations.iter() {
                runtime.push(brain_id, *key, val.clone()).expect(&format!(
                    "Could not push to runtime key: {}, val: {:?}",
                    key, val
                ));
            }
        }
    }
}

/// Add all the brains in `onnx_paths` to a runtime for `brain_repetitions` times and execute
/// all of them once. The time it takes is returned.
/// If `threaded` is true, the runtime is multithreaded, otherwise it is single threaded.
/// The `batch_size` is the number of observations per brain.
fn run_one_shot(onnx_paths: &[&str], brain_repetitions: usize, batch_size: usize, threaded: bool) -> Duration {
    let mut runtime = Runtime::new();
    add_inferers_to_runtime(&mut runtime, &onnx_paths, brain_repetitions, batch_size);
    let start_time = Instant::now();
    if threaded {
        let _ = runtime.run_threaded();
    } else {
        let _ = runtime.run_non_threaded();
    };
    let elapsed_time = start_time.elapsed();
    elapsed_time
}

/// Run the runtime for `duration` seconds and count the number of runs.
/// If `threaded` is true, the runtime is multithreaded, otherwise it is single threaded.	
/// The `batch_size` is the number of observations per brain.
/// The number of runs is returned.
fn run_for(threaded: bool, onnx_paths: &[&str], duration: Duration, batch_size: usize) -> usize {
    let mut runtime = Runtime::new();
    add_inferers_to_runtime(&mut runtime, &onnx_paths, 1000, batch_size);
    // Do a cold run
    runtime.run_threaded();
    push_tickets(&mut runtime, batch_size);

    // Do the actual run
    let result: HashMap<BrainId, HashMap<AgentId, Response<'_>>> = if threaded {
        runtime.run_for_threaded(duration).unwrap()
    } else {
        runtime.run_for_non_threaded(duration).unwrap()
    };
    println!("Result len for threaded is {}", result.len());
    result.len()
}

/// Compare the time it takes to run a single inference for `brain_repetitions` times the brains in
/// `onnx_paths` for `batch_size` observations. 
/// Outputs the speedup obtained by using threading by comparing the time it takes to run the
/// non-threaded version to the threaded version.
fn compare_one_shot(onnx_paths: &[&str], brain_repetitions: usize, batch_size: usize) {
    let non_threaded_time = run_one_shot(&onnx_paths, brain_repetitions, batch_size, false);
    let threaded_time = run_one_shot(&onnx_paths, brain_repetitions, batch_size, true);
    let ratio: f32 = non_threaded_time.as_nanos() as f32 / threaded_time.as_nanos() as f32;
    println!("Batch size is {}", batch_size);
    println!(
        "For {} brain_repetitions, threaded is {} times faster than non threaded",
        brain_repetitions, ratio
    );
    println!("Threaded time: {:?}", threaded_time.as_secs_f64());
    println!("Non Threaded time: {:?}", non_threaded_time.as_secs_f64());
    println!("----")
}

/// Compare the number of runs obtained in `duration` seconds for `brain_repetitions` times the brains in
/// `onnx_paths` for `batch_size` observations. 
/// Outputs the speedup obtained by using threading by comparing the number of runs of the
/// non-threaded version to the threaded version.
fn compare_run_for(onnx_paths: &[&str], duration: Duration, batch_size: usize) {
    let non_threaded_runs = run_for(false, &onnx_paths, duration, batch_size);
    let threaded_runs = run_for(true, &onnx_paths, duration, batch_size);
    let ratio: f32 = threaded_runs as f32 / non_threaded_runs as f32;
    println!(
        "For {} seconds, threaded is {} times more efficient than non threaded",
        duration.as_secs_f32(),
        ratio
    );
    println!("Threaded runs: {:?}", threaded_runs);
    println!("Non Threaded runs: {:?}", non_threaded_runs);
    println!("----")
}

/// Measures the speedup obtained by using threading for the runtime.
pub(crate) fn compare_threading() {
    let brain_repetition_values = [1, 10, 20, 50, 100];
    let batch_sizes = [2, 4, 8, 16, 32, 64];
    println!("One-shot run tests (running through all models once)");
    println!(" ");
    for brain_repetitions in brain_repetition_values {
        for batch_size in batch_sizes {
            println!("-------------------------------");
            println!("Homogenous (same model n times)");
            let onnx_paths = vec!["../../brains/test.onnx"];
            compare_one_shot(&onnx_paths, brain_repetitions, batch_size);

            println!("-------------------------------");
            println!("Heterogeneous (different models once)");
            let onnx_paths = vec![
                "../../brains/test.onnx",
                "../../brains/test-large.onnx",
                "../../brains/test-complex.onnx",
            ];
            compare_one_shot(&onnx_paths, brain_repetitions, batch_size);
        }
    }

    println!("Run for tests (running as many times as possible in a given time)");
    println!(" ");
    for batch_size in batch_sizes {
        println!("Batch size is {}", batch_size);
        println!("-------------------------------");
        println!("Homogenous (same model n times)");
        let onnx_paths = vec!["../../brains/test.onnx"];
        let duration = Duration::from_millis(5);
        compare_run_for(&onnx_paths, duration, batch_size);

        println!("-------------------------------");
        println!("Heterogeneous (different models once)");
        let onnx_paths = vec![
            "../../brains/test.onnx",
            "../../brains/test-large.onnx",
            "../../brains/test-complex.onnx",
        ];
        compare_run_for(&onnx_paths, duration, batch_size);
        break;
    }
}
