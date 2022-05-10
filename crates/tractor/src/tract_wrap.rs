#![allow(clippy::explicit_counter_loop)]

use super::inferer::{Inferer, Observation, Response};
use anyhow::{bail, Error};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

use tract_core::{
    ndarray::{Dim, IntoDimension},
    prelude::*,
};
use tract_hir::{infer::Factoid, prelude::*, shapefactoid};

pub struct TractInstance {
    plan_single: TypedSimplePlan<TypedModel>,
    plan_batched: TypedSimplePlan<TypedModel>,

    normal_distribution: Normal<f32>,
    count: usize,
    batch_size: usize,

    inputs: Vec<(String, Vec<usize>)>,
    outputs: Vec<(String, Vec<usize>)>,
}

pub fn create_plan_with_batchsize(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
) -> TractResult<TypedSimplePlan<TypedModel>> {
    let s = Symbol::new('N');
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(s.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    // optimize the model and get an execution plan
    let model = model.into_optimized()?;

    let plan = SimplePlan::new(model)?;
    Ok(plan)
}

impl TractInstance {
    pub fn from_model(model: InferenceModel, batch_size: usize) -> TractResult<Self> {
        let mut inputs: Vec<(String, Vec<usize>)> = Default::default();

        for input_outlet in model.input_outlets()? {
            let node = model.node(input_outlet.node);
            let name = node.name.split(':').next().unwrap().to_owned();
            let input_shape = &model.input_fact(input_outlet.node)?.shape;

            inputs.push((
                name,
                input_shape
                    .dims()
                    .filter_map(|value| value.concretize().map(|v| v.to_i64().unwrap() as usize))
                    .collect(),
            ));
        }

        let mut outputs: Vec<(String, Vec<usize>)> = Default::default();

        for output_outlet in &model.outputs {
            let name = model.outlet_labels[output_outlet]
                .split(':')
                .next()
                .unwrap()
                .to_owned();

            let output_shape = &model.output_fact(output_outlet.slot)?.shape;

            outputs.push((
                name,
                output_shape
                    .dims()
                    .filter_map(|value| value.concretize().map(|v| v.to_i64().unwrap() as usize))
                    .collect(),
            ));
        }

        let plan_single = create_plan_with_batchsize(model.clone(), &inputs)?;

        let plan_batched = create_plan_with_batchsize(model.clone(), &inputs)?;

        Ok(Self {
            plan_single,
            plan_batched,

            batch_size,
            normal_distribution: Normal::new(0.0, 1.0).unwrap(),
            count: 0,
            inputs,
            outputs,
        })
    }

    fn build_inputs(&mut self, obs: Vec<Observation>) -> (TVec<Tensor>, usize) {
        let size = obs.len();
        let mut inputs = TVec::default();
        let mut named_inputs = TVec::default();

        for (name, shape) in self.inputs.iter() {
            if name == "epsilon" {
                let full_shape = tvec![size, shape.iter().product()];

                named_inputs.push((name, (full_shape, vec![])));
            } else {
                let mut full_shape = tvec![size];
                full_shape.extend_from_slice(&shape);
                let total_count = full_shape.iter().product();
                named_inputs.push((name, (full_shape, Vec::with_capacity(total_count))));
            }
        }

        for observation in obs {
            for (name, (_, store)) in named_inputs.iter_mut() {
                if *name == "epsilon" {
                    continue;
                }
                store.extend_from_slice(&observation.data[*name]);
            }
        }

        for (name, (shape, store)) in named_inputs {
            if name == "epsilon" {
                // Fill epsilon with normal noise
                let mut rng = rand::thread_rng();
                let input1: Tensor =
                    tract_ndarray::Array2::from_shape_fn((size, shape[1]), |(_, _)| {
                        self.normal_distribution.sample(&mut rng)
                    })
                    .into();
                inputs.push(input1);
            } else {
                let tensor = unsafe {
                    tract_ndarray::Array::from_shape_vec_unchecked(shape.into_vec(), store).into()
                };

                inputs.push(tensor);
            }
        }

        (inputs, size)
    }

    pub fn infer_single(
        &mut self,
        obs: Vec<Observation>,
        vec_out: &mut Vec<Response>,
    ) -> TractResult<()> {
        let (inputs, count) = self.build_inputs(obs);
        // Run the optimized plan to get actions back!
        let result = self.plan_single.run(inputs)?;

        for _ in 0..count {
            vec_out.push(Response {
                response: Default::default(),
            });
        }

        for (idx, (name, shape)) in self.outputs.iter().enumerate() {
            for (response_idx, value) in result[idx]
                .to_array_view::<f32>()?
                .as_slice()
                .unwrap()
                .chunks(shape.iter().product())
                .map(|value| value.to_vec())
                .enumerate()
            {
                vec_out[response_idx]
                    .response
                    .insert(name.to_owned(), value);
            }
        }

        Ok(())
    }

    pub fn infer_batched(
        &mut self,
        obs: Vec<Observation>,
        vec_out: &mut Vec<Response>,
    ) -> TractResult<()> {
        let (inputs, count) = self.build_inputs(obs);
        // Run the optimized plan to get actions back!
        let result = self.plan_batched.run(inputs)?;

        for _ in 0..count {
            vec_out.push(Response {
                response: Default::default(),
            });
        }

        for (idx, (name, shape)) in self.outputs.iter().enumerate() {
            for (response_idx, value) in result[idx]
                .to_array_view::<f32>()?
                .as_slice()
                .unwrap()
                .chunks(shape.iter().product())
                .map(|value| value.to_vec())
                .enumerate()
            {
                vec_out[response_idx]
                    .response
                    .insert(name.to_owned(), value);
            }
        }

        Ok(())
    }

    pub fn infer_tract(
        &mut self,
        observations: HashMap<u64, Observation>,
    ) -> TractResult<HashMap<u64, Response>> {
        let mut responses: Vec<Response> = vec![Response::default(); observations.len()];
        let (ids, obs): (Vec<_>, Vec<_>) = observations.into_iter().unzip();

        if obs.len() == 1 {
            self.infer_single(obs, &mut responses)?;
        } else {
            self.infer_batched(obs, &mut responses)?;
        }

        let results: HashMap<u64, Response> = ids.into_iter().zip(responses.drain(..)).collect();

        Ok(results)
    }
}

impl Inferer for TractInstance {
    fn infer(
        &mut self,
        observations: HashMap<u64, Observation>,
    ) -> Result<HashMap<u64, Response>, Error> {
        let tract_result = self.infer_tract(observations);
        match tract_result {
            Ok(results) => Ok(results),
            Err(error) => bail!(format!("{:?}", error)),
        }
    }

    fn input_shapes(&self) -> &Vec<(String, Vec<usize>)> {
        &self.inputs
    }

    fn output_shapes(&self) -> &Vec<(String, Vec<usize>)> {
        &self.outputs
    }
}
