use crate::prada::architecture::{PRADAArchitecture, RowAddress};

use super::{BitwiseOperand, BitwiseRow};
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Row Copy to a single destination
    AAPRowCopy(RowAddress, RowAddress),
    /// TRA on given RowAddresses
    AAPTRA(RowAddress, RowAddress, RowAddress),
    /// Negate operand in row
    N(RowAddress),
}

impl Instruction {
    pub fn get_latency_in_ns(&self) -> u64 {
        match self {
            Instruction::N(_) => 35,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Program<'a> {
    pub architecture: &'a PRADAArchitecture,
    pub instructions: Vec<Instruction>,
    pub runtime_estimate: u64,
    pub energy_consumption_estimate: u64,
}

impl<'a> Program<'a> {
    pub fn new(architecture: &'a PRADAArchitecture, instructions: Vec<Instruction>) -> Self {
        Self {
            architecture,
            instructions,
            runtime_estimate: 0,
            energy_consumption_estimate: 0,
        }
    }
}

impl Instruction {
    pub fn used_addresses<'a>(
        &self,
    ) -> impl Iterator<Item = RowAddress> + 'a {
        match self {
            Instruction::AAPRowCopy(from, _) => vec!(*from).into_iter(),
            Instruction::AAPTRA(a, b, c ) => vec!(*a,*b,*c).into_iter(),
            Instruction::N(a) => vec!(*a).into_iter(),
        }
    }

    pub fn input_operands<'a>(
        &self,
    ) -> impl Iterator<Item = RowAddress> + 'a {
        match self {
            Instruction::AAPRowCopy(from, _) => vec!(*from).into_iter(),
            Instruction::AAPTRA(a, b, c ) => vec!(*a,*b,*c).into_iter(),
            Instruction::N(a) => vec!(*a).into_iter(),
        }
    }
}

impl From<BitwiseRow> for BitwiseOperand {
    fn from(value: BitwiseRow) -> Self {
        match value {
            BitwiseRow::T(t) => BitwiseOperand::T(t),
        }
    }
}

impl Display for Program<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for instruction in &self.instructions {
            match instruction {
                Instruction::AAPRowCopy(a, b) => {
                    write!(f, "AAPRowCopy {a}")?;
                    write!(f, " {b}")?;
                    writeln!(f)?;
                }
                Instruction::AAPTRA(a, b, c) => {
                    write!(f, "AAPTRA {a}")?;
                    write!(f, " {b} {c}")?;
                    writeln!(f)?;
                },
                Instruction::N(a) => {
                    write!(f, "N {a}")?;
                    writeln!(f)?;
                },
            }
        }
        Ok(())
    }
}
