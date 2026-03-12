// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 12 May 2022

use std::sync::Arc;
use tract_core::{
    model::{IntoRunnable, TypedModel, TypedSimplePlan},
    prelude::{Symbol, SymbolValues, ToDim},
    tract_data::TractResult,
};
use tract_hir::prelude::{InferenceModel, InferenceModelExt};

pub(super) fn build_symbolic_model(
    mut model: InferenceModel,
) -> TractResult<(Symbol, TypedModel)> {
    let outlets = model.output_outlets().unwrap().len();
    for output in 0..outlets {
        model.set_output_fact(output, Default::default())?;
    }

    let symbol = model.symbols.sym("N");
    for idx in 0..model.input_outlets()?.len() {
        let mut fact = model.input_fact(idx)?.clone();
        fact.shape.set_dim(0, symbol.to_dim());
        model.set_input_fact(idx, fact)?;
    }

    let model = model.into_typed()?.into_decluttered()?;
    Ok((symbol, model))
}

pub(super) fn build_model<D: ToDim>(
    mut model: InferenceModel,
    batch_dim: D,
) -> TractResult<Arc<TypedSimplePlan>> {
    let outlets = model.output_outlets().unwrap().len();
    for output in 0..outlets {
        model.set_output_fact(output, Default::default())?;
    }

    let batch = batch_dim.to_dim();
    for idx in 0..model.input_outlets()?.len() {
        let mut fact = model.input_fact(idx)?.clone();
        fact.shape.set_dim(0, batch.clone());
        model.set_input_fact(idx, fact)?;
    }

    model
        .into_typed()?
        .into_decluttered()?
        .into_optimized()?
        .into_runnable()
}

pub(super) fn build_symbolic_typed(model: &mut TypedModel) -> TractResult<Symbol> {
    model.declutter()?;
    Ok(model.symbols.sym("N"))
}

pub(super) fn build_typed<D: ToDim>(
    model: TypedModel,
    batch_dim: D,
) -> TractResult<Arc<TypedSimplePlan>> {
    let symbol = model.symbols.sym("N");
    let model = model.concretize_dims(
        &SymbolValues::default().with(&symbol, batch_dim.to_dim().to_i64().unwrap()),
    )?;

    model.into_decluttered()?.into_optimized()?.into_runnable()
}
