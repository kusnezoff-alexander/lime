//! Functionality for generating actual program using architecture defined in [`architecture`] by
//! compiling given logic-network (see [`compilation`]) and potentially adding some manual
//! optimizations ([`optimization`])
use super::architecture::RowAddress;
use crate::fc_dram::architecture::{get_subarrayid_from_rowaddr, Instruction, ROW_ID_BITMASK};
use eggmock::Signal;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};


#[derive(Debug, Clone)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    /// Specifies where row-operands should be placed prior to calling this program
    /// (This is a convention which tells the user of this lib where the data should be placed within the DRAM before executing this program)
    /// - NOTE: Signals might have to be placed in several subarrays (REMINDER: movement in btw subarrays is not supported by FCDRAM)
    pub input_row_operands_placement: HashMap<Signal, Vec<RowAddress>>,
    /// Specifies into which rows output-operands will have been placed after the program has run successfully
    pub output_row_operands_placement: HashMap<Signal, RowAddress>,
}

impl Program {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self {
            instructions,
            input_row_operands_placement: HashMap::new(),
            output_row_operands_placement: HashMap::new(),
        }
    }
}

/// Print the generated program in human-readable form
impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let display_row = |row| {
            format!("{}.{}", get_subarrayid_from_rowaddr(row), row & ROW_ID_BITMASK)
        }; // display subarray separately


        let display_rows = |rows: Vec<RowAddress>| {
            let formatted: Vec<String> = rows.iter()
                .map(|&row| format!("{}.{}", get_subarrayid_from_rowaddr(row), row & ROW_ID_BITMASK))
                .collect();

            format!("[{}]", formatted.join(", "))
        }; // display subarray separately

        writeln!(f, "---------------------------------------")?;
        writeln!(f, "Input operand placement:")?;
        for (signal, rows) in &self.input_row_operands_placement {
            writeln!(f, "{:?} in {}", signal, display_rows(rows.to_vec()))?;
        }
        writeln!(f, "---------------------------------------")?;


        for instr in &self.instructions {
            writeln!(f, "{}", instr)?;
        }

        writeln!(f, "---------------------------------------")?;
        writeln!(f, "Output operand placement:")?;
        for (signal, row) in &self.output_row_operands_placement{
            writeln!(f, "{:?} in {}", signal, display_row(*row))?;
        }
        writeln!(f, "---------------------------------------")?;
        Ok(())
    }
}
