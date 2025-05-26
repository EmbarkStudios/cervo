// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios AB, all rights reserved.
// Created: 12 May 2022

use tract_core::{
    model::{TypedModel, TypedSimplePlan},
    prelude::{Symbol, SymbolValues, ToDim},
    tract_data::{tvec, TractResult},
};
use tract_hir::prelude::{Datum, InferenceFact, InferenceModel, InferenceModelExt};

pub(super) fn build_symbolic_model(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
) -> TractResult<(Symbol, TypedModel)> {
    let outlets = model.output_outlets().unwrap().len();
    for output in 0..outlets {
        model.set_output_fact(output, Default::default())?;
    }

    let symbol = model.symbols.sym("N");
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(symbol.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    let model = model.into_typed()?.into_decluttered()?;
    Ok((symbol, model))
}

pub(super) fn build_model<D: ToDim>(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
    batch_dim: D,
) -> TractResult<TypedSimplePlan<TypedModel>> {
    let outlets = model.output_outlets().unwrap().len();
    for output in 0..outlets {
        model.set_output_fact(output, Default::default())?;
    }

    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(batch_dim.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
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
) -> TractResult<TypedSimplePlan<TypedModel>> {
    let symbol = model.symbols.sym("N");
    let model = model.concretize_dims(
        &SymbolValues::default().with(&symbol, batch_dim.to_dim().to_i64().unwrap()),
    )?;

    model.into_decluttered()?.into_optimized()?.into_runnable()
}
