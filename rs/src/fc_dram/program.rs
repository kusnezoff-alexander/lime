//! Functionality for generating actual program using architecture defined in [`architecture`] by
//! compiling given logic-network (see [`compilation`]) and potentially adding some manual
//! optimizations ([`optimization`])
use super::architecture::{FCDRAMArchitecture, RowAddress};
use crate::fc_dram::architecture::Instruction;
use eggmock::{Id, Aig, NetworkWithBackwardEdges, Signal};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};


#[derive(Debug, Clone)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    /// Specifies where row-operands should be placed prior to calling this program
    /// (This is a convention which tells the user of this lib where the data should be placed within the DRAM before executing this program)
    pub input_row_operands_placement: HashMap<Signal, RowAddress>,
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
        writeln!(f, "---------------------------------------")?;
        writeln!(f, "Input operand placement:")?;
        for (signal, row) in &self.input_row_operands_placement {
            writeln!(f, "{:?} in {}", signal, row)?;
        }
        writeln!(f, "---------------------------------------")?;


        for instr in &self.instructions {
            writeln!(f, "{}", instr)?;
        }

        writeln!(f, "---------------------------------------")?;
        writeln!(f, "Output operand placement:")?;
        for (signal, row) in &self.output_row_operands_placement{
            writeln!(f, "{:?} in {}", signal, row)?;
        }
        writeln!(f, "---------------------------------------")?;
        Ok(())
    }
}
