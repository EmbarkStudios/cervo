// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Tom Solberg, all rights reserved.
// Created: 29 July 2022

/*!

*/
use crate::BrainId;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CervoError {
    #[error("unknown brain with id {0:?}")]
    UnknownBrain(BrainId),

    #[error("internal error occured: {0}")]
    Internal(anyhow::Error),
}
