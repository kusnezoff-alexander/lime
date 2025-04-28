use super::{Architecture, BitwiseOperand, BitwiseRow, RowOperand, Rows};
use eggmock::{Id, Mig, ProviderWithBackwardEdges, Signal};
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

/// Instructions which operate on DRAM-Rows
/// TODO: adjust instructions to FC-DRAM?!
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// TODO: What does this instr do?
    /// TODO: which row is operand, which one is a result?
    AAP(RowOperand, RowOperand),
    /// TODO: What does this instr do?
    AP(RowOperand),
    /// Multiple-Row Activation: `ACT R_F -> PRE -> ACT R_L -> PRE` of rows `R_F`,`R_L` for rows within
    /// different subarrays. As a result `R_L` holds the negated value of `R_F`
    /// Used to implement NOT directly
    APAP(RowOperand,RowOperand),
}

#[derive(Debug, Clone)]
pub struct Program<'a> {
    pub architecture: &'a Architecture,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub struct ProgramState<'a> {
    program: Program<'a>,
    /// currently used rows
    rows: Rows<'a>,
}

impl<'a> Program<'a> {
    pub fn new(architecture: &'a Architecture, instructions: Vec<Instruction>) -> Self {
        Self {
            architecture,
            instructions,
        }
    }
}

impl Instruction {
    /// Return Addreses of Rows which are used by this instruction (=operand-rows AND result-row)
    pub fn used_addresses<'a>(
        &self,
        architecture: &'a Architecture,
    ) -> impl Iterator<Item = RowAddress> + 'a {
        let from = match self {
            Instruction::AAP(from, _) => from,
            Instruction::AP(op) => op,
        }
        .row_addresses(architecture);
        let to = match self {
            Instruction::AAP(_, to) => Some(*to),
            _ => None,
        }
        .into_iter()
        .flat_map(|addr| addr.row_addresses(architecture));
        from.chain(to)
    }

    /// Return addresses of operand-rows
    pub fn input_operands<'a>(
        &self,
        architecture: &'a Architecture,
    ) -> impl Iterator<Item = RowAddress> + 'a {
        let from = match self {
            Instruction::AAP(from, _) => from,
            Instruction::AP(op) => op,
        };
        from.row_addresses(architecture)
    }

    pub fn overridden_rows<'a>(
        &self,
        architecture: &'a Architecture,
    ) -> impl Iterator<Item = RowOperand> + 'a {
        let first = match self {
            Instruction::AP(a) => a,
            Instruction::AAP(a, _) => a,
        }
        .clone();
        let first = match first {
            Address::Bitwise(BitwiseAddress::Multiple(idx)) => {
                architecture.multi_activations[idx].as_slice()
            }
            _ => &[],
        }
        .iter()
        .map(|op| RowOperand::Bitwise(op.row()));

        let second = match self {
            Instruction::AP(_) => None,
            Instruction::AAP(_, a) => Some(a.row_addresses(architecture).map(|addr| addr.row())),
        }
        .into_iter()
        .flatten();

        first.chain(second)
    }
}

impl<'a> ProgramState<'a> {
    pub fn new(
        architecture: &'a Architecture,
        network: &impl ProviderWithBackwardEdges<Node = Mig>,
    ) -> Self {
        Self {
            program: Program::new(architecture, Vec::new()),
            rows: Rows::new(network, architecture),
        }
    }

    pub fn maj(&mut self, op: usize, out_signal: Signal, out_address: Option<Address>) {
        let operands = &self.architecture.multi_activations[op];
        for operand in operands {
            self.set_signal(RowAddress::Bitwise(*operand), out_signal);
        }
        let instruction = match out_address {
            Some(out) => Instruction::AAP(BitwiseAddress::Multiple(op).into(), out),
            None => Instruction::AP(BitwiseAddress::Multiple(op).into()),
        };
        self.instructions.push(instruction)
    }

    /// TODO: Does FC-DRAM need copying of signals?
    /// pub fn signal_copy(&mut self, signal: Signal, target: RowAddress, intermediate_dcc: u8) {
    /// }

    /// Sets the value of the operand in `self.rows` to the given signal. If that removes the last
    /// reference to the node of the previous signal of the operator, insert spill code for the
    /// previous signal
    /// **ALWAYS** call this before inserting the actual instruction, otherwise the spill code will
    /// spill the wrong value
    fn set_signal(&mut self, address: RowAddress, signal: Signal) {
        if let Some(previous_signal) = self.rows.set_signal(address, signal) {
            if !self.rows.contains_id(previous_signal.node_id()) {
                let spill_id = self.rows.add_spill(previous_signal);
                self.instructions
                    .push(Instruction::AAP(address.into(), Address::Spill(spill_id)));
            }
        }
    }

    pub fn free_id_rows(&mut self, id: Id) {
        self.rows.free_id_rows(id);
    }

    pub fn rows(&self) -> &Rows {
        &self.rows
    }
}

impl<'a> Deref for ProgramState<'a> {
    type Target = Program<'a>;

    fn deref(&self) -> &Self::Target {
        &self.program
    }
}

impl DerefMut for ProgramState<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.program
    }
}

impl<'a> From<ProgramState<'a>> for Program<'a> {
    fn from(value: ProgramState<'a>) -> Self {
        value.program
    }
}

/// Print the generated program in human-readable form
impl Display for Program<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let write_operand = |f: &mut Formatter<'_>, o: &BitwiseOperand| -> std::fmt::Result {
            match o {
                BitwiseOperand::T(t) => write!(f, "T{}", t),
                BitwiseOperand::DCC { inverted, index } => {
                    if *inverted {
                        write!(f, "~DCC{}", index)
                    } else {
                        write!(f, "DCC{}", index)
                    }
                }
            }
        };
        let write_address = |f: &mut Formatter<'_>, a: &Address| -> std::fmt::Result {
            match a {
                Address::In(i) => write!(f, "I{}", i),
                Address::Out(i) => write!(f, "O{}", i),
                Address::Spill(i) => write!(f, "S{}", i),
                Address::Const(c) => write!(f, "C{}", if *c { "1" } else { "0" }),
                Address::Bitwise(b) => match b {
                    BitwiseAddress::Single(o) => write_operand(f, o),
                    BitwiseAddress::Multiple(id) => {
                        let operands = &self.architecture.multi_activations[*id];
                        for i in 0..operands.len() {
                            if i == 0 {
                                write!(f, "[")?;
                            } else {
                                write!(f, ", ")?;
                            }
                            write_operand(f, &operands[i])?;
                            if i == operands.len() - 1 {
                                write!(f, "]")?;
                            }
                        }
                        Ok(())
                    }
                },
            }
        };

        for instruction in &self.instructions {
            match instruction {
                Instruction::AAP(a, b) => {
                    write!(f, "AAP ")?;
                    write_address(f, a)?;
                    write!(f, " ")?;
                    write_address(f, b)?;
                    writeln!(f)?;
                }
                Instruction::AP(a) => {
                    write!(f, "AP ")?;
                    write_address(f, a)?;
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}
