/*!

# Cervo Core

This crate contains some wrappers and extensions for Tract we use to
simplify our workflows.

*/
pub use tract_core;
pub use tract_hir;

pub mod epsilon;
pub mod inferer;
mod model_api;

/// Most core utilities are re-exported here.
pub mod prelude {
    pub use super::epsilon::{
        EpsilonInjector, HighQualityNoiseGenerator, LowQualityNoiseGenerator, NoiseGenerator,
    };
    pub use super::inferer::{
        BasicInferer, DynamicMemoizingInferer, FixedBatchInferer, Inferer, InfererBuilder,
        InfererExt, InfererProvider, State,
    };
    pub use super::model_api::ModelApi;
}
