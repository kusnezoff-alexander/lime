use crate::prada::{architecture::{PRADAArchitecture, RowAddress}, BitwiseOperand};

use eggmock::{Id, Mig, NetworkWithBackwardEdges, Signal};
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

/// Equivalent to either a DCC row or a T row in the bitwise row section. DCC row always refers to
/// the non-inverted row.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BitwiseRow {
    T(u8),
}
