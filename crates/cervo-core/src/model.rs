use crate::batcher::ScratchPadView;
use crate::inferer::Inferer;
pub trait ModelWrapper {
    fn input_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)];
    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)];
    fn invoke(&self, inferer: &impl Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()>;
    fn begin_agent(&self, id: u64);
    fn end_agent(&self, id: u64);
}

pub struct BaseCase;

impl ModelWrapper for BaseCase {
    fn input_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        inferer.input_shapes()
    }

    fn output_shapes<'a>(&'a self, inferer: &'a dyn Inferer) -> &'a [(String, Vec<usize>)] {
        inferer.output_shapes()
    }

    fn invoke(&self, inferer: &impl Inferer, batch: &mut ScratchPadView<'_>) -> anyhow::Result<()> {
        inferer.infer_raw(batch)
    }

    fn begin_agent(&self, id: u64) {}
    fn end_agent(&self, id: u64) {}
}

pub struct Model<WrapStack: ModelWrapper, Policy: Inferer> {
    wrapper_stack: WrapStack,
    policy: Policy,
}

impl<WrapStack: ModelWrapper, Policy: Inferer> Model<WrapStack, Policy> {
    pub fn new(wrapper_stack: WrapStack, policy: Policy) -> Self {
        Self {
            wrapper_stack,
            policy,
        }
    }

    /// Replace the inner policy with a new policy while maintaining any state in wrappers.
    ///
    /// Requires that the shapes of the policies are compatible, but
    /// they may be different concrete inferer implementations.
    pub fn with_new_policy<NewPolicy: Inferer>(
        self,
        new_policy: NewPolicy,
    ) -> Result<Model<WrapStack, NewPolicy>, (Self, anyhow::Error)> {
        if let Err(e) = Self::check_compatible_shapes(&self.policy, &new_policy) {
            return Err((self, e));
        }
        Ok(Model {
            wrapper_stack: self.wrapper_stack,
            policy: new_policy,
        })
    }

    /// Validate that [`Old`] and [`New`] are compatible with each
    /// other.
    fn check_compatible_shapes<Old: Inferer, New: Inferer>(
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

    pub fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.wrapper_stack.input_shapes(&self.policy)
    }

    pub fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.wrapper_stack.output_shapes(&self.policy)
    }
}

impl<WrapStack: ModelWrapper, Policy: Inferer> Inferer for Model<WrapStack, Policy> {
    fn select_batch_size(&self, max_count: usize) -> usize {
        self.policy.select_batch_size(max_count)
    }

    fn infer_raw(&self, batch: &mut ScratchPadView<'_>) -> anyhow::Result<(), anyhow::Error> {
        self.wrapper_stack.invoke(&self.policy, batch)
    }

    fn raw_input_shapes(&self) -> &[(String, Vec<usize>)] {
        self.policy.raw_input_shapes()
    }

    fn raw_output_shapes(&self) -> &[(String, Vec<usize>)] {
        self.policy.raw_output_shapes()
    }

    fn begin_agent(&mut self, id: u64) {
        self.wrapper_stack.begin_agent(id)
    }

    fn end_agent(&mut self, id: u64) {
        self.wrapper_stack.end_agent(id)
    }
}
