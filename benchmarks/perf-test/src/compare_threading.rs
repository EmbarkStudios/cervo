use cervo_core::inferer::InfererExt;
use cervo_core::prelude::Inferer;
use cervo_core::prelude::LowQualityNoiseGenerator;
use cervo_core::prelude::State;
use cervo_runtime::BrainId;
use cervo_runtime::Runtime;
use std::time::Duration;
use std::time::Instant;

// TODO: Luc: Experiment with batches of 2, 4, 8, 16, 32, 64. 
// TODO: Luc: For run for, let it do a cold run first, then do the actual run to let the models determine the time it takes to run

fn add_inferers_to_runtime(runtime: &mut Runtime, onnx_paths: &[&str], runs: usize) {
    let batch_size = 32;
    for i in 0..runs {
        for onnx_path in onnx_paths {
            let mut reader = crate::helpers::get_file(onnx_path).expect("Could not open file");
            let mut inferer = cervo_onnx::builder(&mut reader)
                .build_fixed(&[batch_size])
                // .build_dynamic()
                // .unwrap()
                // .with_epsilon(LowQualityNoiseGenerator::default(), "epsilon")
                .unwrap();


            let inputs = inferer.input_shapes().to_vec();
            let observations = crate::helpers::build_inputs_from_desc(batch_size as u64, &inputs);
            runtime.add_inferer(inferer);
            
            for (key, val) in observations.iter() {
                runtime
                    .push(BrainId(i as u16), *key, val.clone())
                    .expect(&format!("Could not push to runtime key: {}, val: {:?}", key, val));
            }

        }
    }
}

fn threaded_one_shot(onnx_paths: &[&str], runs: usize) -> Duration {
    let mut runtime = Runtime::new();
    add_inferers_to_runtime(&mut runtime, &onnx_paths, runs);

    let start_time = Instant::now();
    runtime.run_threaded();
    let elapsed_time = start_time.elapsed();
    elapsed_time
}

fn non_threaded_one_shot(onnx_paths: &[&str], runs: usize) -> Duration {
    let mut runtime = Runtime::new();
    add_inferers_to_runtime(&mut runtime, &onnx_paths, runs);

    let start_time = Instant::now();
    runtime.run_non_threaded();
    let elapsed_time = start_time.elapsed();
    elapsed_time
}

fn threaded_run_for(onnx_paths: &[&str], duration: Duration) -> usize {
    let mut runtime = Runtime::new();
    add_inferers_to_runtime(&mut runtime, &onnx_paths, 1000);

    let previous_len = runtime.queue_len();
    runtime.run_for_threaded(duration);
    let current_len = runtime.queue_len();
    previous_len - current_len
}


fn non_threaded_run_for(onnx_paths: &[&str], duration: Duration) -> usize {
    let mut runtime = Runtime::new();
    // Do a cold run
    add_inferers_to_runtime(&mut runtime, &onnx_paths, 500);
    runtime.run_for_non_threaded(duration);

    // Do the actual run

    let previous_len = runtime.queue_len();
    println!("Previous Queue len is {}", previous_len);
    runtime.run_for_non_threaded(duration);
    let current_len = runtime.queue_len();
    println!("Current Queue len is {}", current_len);
    previous_len - current_len
}

fn compare_one_shot(onnx_paths: &[&str], runs: usize) {
    let non_threaded_time = non_threaded_one_shot(&onnx_paths, runs);
    let threaded_time = threaded_one_shot(&onnx_paths, runs);
    let ratio: f32 = non_threaded_time.as_nanos() as f32 / threaded_time.as_nanos() as f32;
    println!(
        "For {} runs, threaded is {} times faster than non threaded",
        runs, ratio
    );
    println!("Threaded time: {:?}", threaded_time.as_secs_f64());
    println!("Non Threaded time: {:?}", non_threaded_time.as_secs_f64());
    println!("----")
}

fn compare_run_for(onnx_paths: &[&str], duration: Duration) {
    let non_threaded_runs = non_threaded_run_for(&onnx_paths, duration);
    let threaded_runs = threaded_run_for(&onnx_paths, duration);
    let ratio: f32 = non_threaded_runs as f32 / threaded_runs as f32;
    println!(
        "For {} seconds, threaded is {} times more efficient than non threaded",
        duration.as_secs_f32(),
        ratio
    );
    println!("Threaded runs: {:?}", threaded_runs);
    println!("Non Threaded runs: {:?}", non_threaded_runs);
    println!("----")
}

pub(crate) fn compare_threading() {
    println!("One-shot run tests (running through all models once)");
    println!(" ");
    {
        println!("-------------------------------");
        println!("Homogenous (same model n times)");
        let onnx_paths = vec!["../../brains/test.onnx"];
        for runs in [10, 100, 500, 1000] {
            compare_one_shot(&onnx_paths, runs);
        }

        println!("-------------------------------");
        println!("Heterogeneous (different models once)");
        let onnx_paths = vec![
            "../../brains/test.onnx",
            // "../../brains/test-large.onnx",
            // "../../brains/test-complex.onnx",
        ];
        compare_one_shot(&onnx_paths, 1);
    }

    // println!("Run for tests (running as many times as possible in a given time)");
    // println!(" ");
    // {
    //     println!("-------------------------------");
    //     println!("Homogenous (same model n times)");
    //     let onnx_paths = vec!["../../brains/test.onnx"];
    //     let duration = Duration::from_secs(5);
    //     compare_run_for(&onnx_paths, duration);

    //     println!("-------------------------------");
    //     println!("Heterogeneous (different models once)");
    //     let onnx_paths = vec!["../../brains/test.onnx", "../../brains/test-large.onnx", "../../brains/test-complex.onnx"];
    //     compare_run_for(&onnx_paths, duration);

    // }
}

// Run
//     Heterogeneous (different models once)
//         Threaded
//         Non threaded
//     Homogenous (same model 10 times)
//         Threaded
//         Non threaded

// Run For
//     Heterogeneous (different models once)
//         Threaded
//         Non threaded
//     Homogenous (same model 10 times)
//         Threaded
//         Non threaded
