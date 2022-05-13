// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 12 May 2022

/*!

*/

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
    let s = Symbol::from('N');
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(s.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    let model = model.into_typed()?.into_decluttered()?;
    Ok((s, model))
}

pub(super) fn build_symbolic_typed(model: &mut TypedModel) -> TractResult<Symbol> {
    let s = Symbol::from('N');
    model.declutter()?;
    Ok(s)
}

pub(super) fn build_model<D: ToDim>(
    mut model: InferenceModel,
    inputs: &[(String, Vec<usize>)],
    batch_dim: D,
) -> TractResult<TypedSimplePlan<TypedModel>> {
    for (idx, (_name, shape)) in inputs.iter().enumerate() {
        let mut full_shape = tvec!(batch_dim.to_dim());

        full_shape.extend(shape.iter().map(|v| (*v as i32).into()));
        model.set_input_fact(idx, InferenceFact::dt_shape(f32::datum_type(), full_shape))?;
    }

    let model = model
        .into_optimized()?
        .into_decluttered()?
        .into_runnable()?;

    Ok(model)
}

pub(super) fn build_typed<D: ToDim>(
    model: TypedModel,
    batch_dim: D,
) -> TractResult<TypedSimplePlan<TypedModel>> {
    let symbol = Symbol::from('N');
    let model = model.concretize_dims(
        &SymbolValues::default().with(symbol, batch_dim.to_dim().to_i64().unwrap()),
    )?;

    let model = model.into_optimized()?.into_runnable()?;

    Ok(model)
}
