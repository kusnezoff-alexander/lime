use std::{fmt::{Debug, Display, Formatter, Result}, sync::LazyLock};

pub const NR_SUBARRAYS: u64 = 2u64.pow(7);
pub const ROWS_PER_SUBARRAY: u64 = 2u64.pow(9);
pub const SUBARRAY_ID_BITMASK: u64 = 0b1_111_111_000_000_000; // 7 highest bits=subarray id
pub const ROW_ID_BITMASK: u64 = 0b0_000_000_111_111_111; // 7 highest bits=subarray id

// some utility functions
pub fn subarrayid_to_subarray_address(subarray_id: SubarrayId) -> RowAddress {
    RowAddress(subarray_id.0 << ROWS_PER_SUBARRAY.ilog2()) // lower bits=rows in subarray
}

#[derive(Clone, Debug)]
pub struct PRADAArchitecture {
    /// Nr of subarrays in a DRAM module
    pub nr_subarrays: u64,
    /// Nr of rows in a single subarray
    pub rows_per_subarray: u64,
}

impl PRADAArchitecture {
    pub fn new(nr_subarrays: u64, rows_per_subarray: u64) -> Self {
        Self {
            nr_subarrays,
            rows_per_subarray,
        }
    }

}


/// Main variable specifying architecture of DRAM-module for which to compile for
/// - this is currently just an example implementation for testing purpose; (TODO: make this configurable at runtime)
pub static ARCHITECTURE: LazyLock<PRADAArchitecture> = LazyLock::new(|| {

    PRADAArchitecture {
        nr_subarrays: NR_SUBARRAYS,
        rows_per_subarray: ROWS_PER_SUBARRAY,
    }
});

/// - ! must be smaller than `rows_per_subarray * nr_subarrays` (this is NOT checked!)
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct RowAddress(pub u64);

impl RowAddress {
    /// Return subarray-id the row lies in
    pub fn get_subarray_id(&self) -> SubarrayId {
        SubarrayId((self.0 & SUBARRAY_ID_BITMASK) >> ROWS_PER_SUBARRAY.ilog2())
    }

    /// Converts RowAddress to the same row address but in the other subarray
    pub fn local_rowaddress_to_subarray_id(&self, subarray_id: SubarrayId) -> RowAddress {
        let local_row_address = self.0 & ROW_ID_BITMASK;
        RowAddress( local_row_address | subarrayid_to_subarray_address(subarray_id).0 )
    }
}

impl Display for RowAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}.{}", self.get_subarray_id().0, self.0 & ROW_ID_BITMASK)
    }
}

impl Debug for RowAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}.{}", self.get_subarray_id().0, self.0 & ROW_ID_BITMASK)
    }
}

impl From<u64> for RowAddress {
    fn from(value: u64) -> Self {
        RowAddress(value)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct SubarrayId(pub u64);

impl SubarrayId {
    /// Currently all ops only work on half of the cells (every 2nd cell) with two subarrays being
    /// in a compute/reference subarray "relation" with exactly one other neighboring subarray.
    /// This function returns that other partner (compute/reference) subarray
    pub fn get_partner_subarray(&self) -> Self {
        if self.0.is_multiple_of(2) {
            SubarrayId(self.0 + 1)
        } else {
            SubarrayId(self.0 - 1)
        }
    }
}

impl Display for SubarrayId {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.0)
    }
}
