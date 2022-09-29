// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 29 July 2022

/*!

*/
use crate::BrainId;
use thiserror::Error;

/// Errors that can be returned by Cervo Runtime.
#[derive(Error, Debug)]
pub enum CervoError {
    #[error("unknown brain with id {0:?}")]
    UnknownBrain(BrainId),

    #[error("the runtime was cleared but the following brains still had data: {0:?}")]
    OrphanedData(Vec<BrainId>),

    #[error("internal error occured: {0}")]
    Internal(anyhow::Error),
}
