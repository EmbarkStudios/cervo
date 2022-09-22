use std::cmp::Ordering;

use crate::BrainId;

/// A ticket for the ML in the queue for execution.
pub(super) struct Ticket(pub(super) u64, pub(super) BrainId);

impl PartialEq for Ticket {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Ticket {}

impl PartialOrd for Ticket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for Ticket {
    fn cmp(&self, other: &Ticket) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
