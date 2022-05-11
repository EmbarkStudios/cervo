// Author: Tom Olsson <tom.olsson@embark-studios.com>
// Copyright © 2020, Embark Studios, all rights reserved.
// Created: 18 May 2020

#![warn(clippy::all)]

use anyhow::Error;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct State {
    pub data: HashMap<String, Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct Response {
    pub response: HashMap<String, Vec<f32>>,
}

#[derive(Clone)]
pub struct ModelData {
    pub data: Vec<u8>,
}

pub trait Inferer {
    fn infer(&mut self, observations: HashMap<u64, State>)
        -> Result<HashMap<u64, Response>, Error>;

    fn input_shapes(&self) -> &[(String, Vec<usize>)];
    fn output_shapes(&self) -> &[(String, Vec<usize>)];
}
