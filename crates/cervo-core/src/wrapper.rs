/*!
Inferer wrappers with state separated from the inferer.

This allows separation of stateful logic from the inner inferer,
allowing the inner inferer to be swapped out while maintaining
state in the wrappers.

This is an alternative to the old layered inferer setup, which
tightly coupled the inner inferer with the wrapper state.

```rust,ignore
let inferer = ...;
// the root needs [`BaseCase`] passed as a base case.
let wrappers = RecurrentTrackerWrapper::new(BaseCase, inferer);
let wrapped = StatefulInferer::new(wrappers, infere);
// or
let wrapped = inferer.into_stateful(wrappers);
// or
let wrapped = wrappers.wrap(inferer);
```
*/

use crate::batcher::ScratchPadView;
use crate::inferer::{
    BasicInferer, DynamicInferer, FixedBatchInferer, Inferer, MemoizingDynamicInferer,
};

/// A trait for wrapping an inferer with additional functionality.
///
/// This works similar to the old layered inferer setup, but allows
/// separation of wrapper state from the inner inferer. This allows
/// swapping out the inner inferer while maintaining state in the
/// wrappers.
pub trait InfererWrapper {
    /// Returns the input shapes after this wrapper has been applied.
    fn input_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)];

    /// Returns the output shapes after this wrapper has been applied.
    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)];

    /// Invokes the inner inferer, applying any additional logic before
    /// and after the call.
    fn invoke(&self, inferer: &dyn Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()>;

    /// Called when starting inference for a new agent.
    fn begin_agent(&self, inferer: &dyn Inferer, id: u64);

    /// Called when finishing inference for an agent.
    fn end_agent(&self, inferer: &dyn Inferer, id: u64);
}

/// A no-op inferer wrapper that just calls the inner inferer directly. This is the base-case of wrapper stack.
pub struct BaseWrapper;

impl InfererWrapper for BaseWrapper {
    /// Returns the input shapes after this wrapper has been applied.
    fn input_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        inferer.input_shapes()
    }

    /// Returns the output shapes after this wrapper has been applied.
    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        inferer.output_shapes()
    }

    /// Invokes the inner inferer.
    fn invoke(&self, inferer: &dyn Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()> {
        inferer.infer_raw(batch)
    }

    fn begin_agent(&self, inferer: &dyn Inferer, id: u64) {
        inferer.begin_agent(id);
    }

    fn end_agent(&self, inferer: &dyn Inferer, id: u64) {
        inferer.end_agent(id);
    }
}

impl InfererWrapper for Box<dyn InfererWrapper> {
    fn input_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        self.as_ref().input_shapes(inferer)
    }

    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        self.as_ref().output_shapes(inferer)
    }

    fn invoke(&self, inferer: &dyn Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()> {
        self.as_ref().invoke(inferer, batch)
    }

    fn begin_agent(&self, inferer: &dyn Inferer, id: u64) {
        self.as_ref().begin_agent(inferer, id);
    }

    fn end_agent(&self, inferer: &dyn Inferer, id: u64) {
        self.as_ref().end_agent(inferer, id);
    }
}

/// An inferer that maintains state in wrappers around an inferer.
///
/// This is an alternative to direct wrapping of an inferer, which
/// allows the inner inferer to be swapped out while maintaining
/// state in the wrappers.
pub struct StatefulInferer<WrapStack: InfererWrapper, Inf: Inferer> {
    wrapper_stack: WrapStack,
    inferer: Inf,
}

impl<WrapStack: InfererWrapper, Inf: Inferer> StatefulInferer<WrapStack, Inf> {
    /// Construct a new [`StatefulInferer`] by wrapping the given
    /// inferer with the given wrapper stack.
    pub fn new(wrapper_stack: WrapStack, inferer: Inf) -> Self {
        Self {
            wrapper_stack,
            inferer,
        }
    }

    /// Replace the inner inferer with a new inferer while maintaining
    /// any state in wrappers.
    ///
    /// Requires that the shapes of the policies are compatible, but
    /// they may be different inferer types. If this check fails, will
    /// return self unchanged.
    pub fn with_new_inferer<NewInf: Inferer>(
        self,
        new_inferer: NewInf,
    ) -> Result<StatefulInferer<WrapStack, NewInf>, (Self, anyhow::Error)> {
        if let Err(e) = Self::check_compatible_shapes(&self.inferer, &new_inferer) {
            return Err((self, e));
        }
        Ok(StatefulInferer {
            wrapper_stack: self.wrapper_stack,
            inferer: new_inferer,
        })
    }

    /// Replace the inner inferer with a new inferer while maintaining
    /// any state in wrappers.
    ///
    /// Requires that the shapes of the policies are compatible If
    /// this check fails, will not change self. Compared to
    /// [`with_new_inferer`], also requires that the new inferer has
    /// the same type as the old one.
    pub fn replace_inferer(&mut self, new_inferer: Inf) -> anyhow::Result<()> {
        if let Err(e) = Self::check_compatible_shapes(&self.inferer, &new_inferer) {
            Err(e)
        } else {
            self.inferer = new_inferer;
            Ok(())
        }
    }

    /// Validate that [`Old`] and [`New`] are compatible with each
    /// other.
    pub fn check_compatible_shapes<Old: Inferer, New: Inferer>(
        old: &Old,
        new: &New,
    ) -> Result<(), anyhow::Error> {
        let old_in = old.raw_input_shapes();
        let new_in = new.raw_input_shapes();

        let old_out = old.raw_output_shapes();
        let new_out = new.raw_output_shapes();

        for (i, (o, n)) in old_in.iter().zip(new_in).enumerate() {
            if o != n {
                if o.0 != n.0 {
                    return Err(anyhow::format_err!(
                        "name mismatch for input {i}: '{}' != '{}'",
                        o.0,
                        n.0,
                    ));
                }

                return Err(anyhow::format_err!(
                    "shape mismatch for input '{}': {:?} != {:?}",
                    o.0,
                    o.1,
                    n.1,
                ));
            }
        }

        for (i, (o, n)) in old_out.iter().zip(new_out).enumerate() {
            if o != n {
                if o.0 != n.0 {
                    return Err(anyhow::format_err!(
                        "name mismatch for output {i}: '{}' != '{}'",
                        o.0,
                        n.0,
                    ));
                }

                return Err(anyhow::format_err!(
                    "shape mismatch for output {}: {:?} != {:?}",
                    o.0,
                    o.1,
                    n.1,
                ));
            }
        }

        Ok(())
    }

    /// Returns the input shapes after all wrappers have been applied.
    pub fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.wrapper_stack.input_shapes(&self.inferer)
    }

    /// Returns the output shapes after all wrappers have been applied.
    pub fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.wrapper_stack.output_shapes(&self.inferer)
    }
}

/// See [`Inferer`] for documentation.
impl<WrapStack: InfererWrapper, Inf: Inferer> Inferer for StatefulInferer<WrapStack, Inf> {
    fn select_batch_size(&self, max_count: usize) -> usize {
        self.inferer.select_batch_size(max_count)
    }

    fn infer_raw(&self, batch: &mut ScratchPadView<'_>) -> anyhow::Result<(), anyhow::Error> {
        self.wrapper_stack.invoke(&self.inferer, batch)
    }

    fn raw_input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inferer.raw_input_shapes()
    }

    fn raw_output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.inferer.raw_output_shapes()
    }

    fn begin_agent(&self, id: u64) {
        self.wrapper_stack.begin_agent(&self.inferer, id);
    }

    fn end_agent(&self, id: u64) {
        self.wrapper_stack.end_agent(&self.inferer, id);
    }
}

/// Extension trait to allow easy wrapping of an inferer with a wrapper stack.
pub trait IntoStateful: Inferer + Sized {
    /// Construct a [`StatefulInferer`] by wrapping this concrete
    /// inferer with the given wrapper stack.
    fn into_stateful<WrapStack: InfererWrapper>(
        self,
        wrapper_stack: WrapStack,
    ) -> StatefulInferer<WrapStack, Self> {
        StatefulInferer::new(wrapper_stack, self)
    }
}

impl IntoStateful for BasicInferer {}
impl IntoStateful for DynamicInferer {}
impl IntoStateful for MemoizingDynamicInferer {}
impl IntoStateful for FixedBatchInferer {}

/// Extension trait to allow easy wrapping of an inferer with a wrapper stack.
pub trait InfererWrapperExt: InfererWrapper + Sized {
    /// Construct a [`StatefulInferer`] by wrapping an inner inferer with this wrapper.
    fn wrap<Inf: Inferer>(self, inferer: Inf) -> StatefulInferer<Self, Inf> {
        StatefulInferer::new(self, inferer)
    }
}

impl<T: InfererWrapper> InfererWrapperExt for T {}
