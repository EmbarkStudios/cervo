use std::cmp::Ordering;

use crate::BrainId;

/// A ticket for the ML in the queue for execution.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ticket(pub(super) u64, pub(super) BrainId);

// SAFETY: Ticket is only ever accessed from a single thread, and is never shared between threads
#[allow(unsafe_code)]
unsafe impl Sync for Ticket {}

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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::{BrainId, Ticket};
    #[test]
    fn ticket_cmp_is_inversed() {
        let a = Ticket(0, BrainId(0));
        let b = Ticket(1, BrainId(1));

        // a has lower sequence number = higher sorting power
        assert_eq!(a.cmp(&b), Ordering::Greater);
    }

    #[test]
    fn ticket_cmp_ignore_brain() {
        let a = Ticket(0, BrainId(1));
        let b = Ticket(1, BrainId(0));

        // a has lower sequence number = higher sorting power
        assert_eq!(a.cmp(&b), Ordering::Greater);
    }

    #[test]
    fn ticket_cmp_is_inversed_reverse_cmp() {
        let a = Ticket(0, BrainId(0));
        let b = Ticket(1, BrainId(1));

        // a has lower sequence number = higher sorting power, so b is less.
        assert_eq!(b.cmp(&a), Ordering::Less);
    }
}
