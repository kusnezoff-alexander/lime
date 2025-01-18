use super::{Architecture, BitwiseOperand, BitwiseRow, Row, Rows};
use eggmock::{MigNode, ProviderWithBackwardEdges, Signal};
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Address {
    In(u64),
    Spill(u32),
    Const(bool),
    Bitwise(BitwiseAddress),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SingleRowAddress {
    In(u64),
    Spill(u32),
    Const(bool),
    Bitwise(BitwiseOperand),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitwiseAddress {
    Single(BitwiseOperand),
    Maj(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    AAP(Address, Address),
    AP(Address),
}

#[derive(Debug, Clone)]
pub struct Program<'a> {
    pub architecture: &'a Architecture,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub struct ProgramState<'a> {
    program: Program<'a>,
    rows: Rows,
}

impl<'a> Program<'a> {
    pub fn new(architecture: &'a Architecture, instructions: Vec<Instruction>) -> Self {
        Self {
            architecture,
            instructions,
        }
    }
}

impl<'a> ProgramState<'a> {
    pub fn new(
        architecture: &'a Architecture,
        network: &impl ProviderWithBackwardEdges<Node = MigNode>,
    ) -> Self {
        Self {
            program: Program::new(architecture, Vec::new()),
            rows: Rows::new(network),
        }
    }

    pub fn maj(&mut self, op: usize, out_signal: Signal) {
        let operands = self.architecture.maj_ops[op];
        for operand in operands {
            self.set_operand_signal(operand, out_signal);
        }
        self.instructions
            .push(Instruction::AP(BitwiseAddress::Maj(op).into()));
    }

    pub fn signal_copy(&mut self, signal: Signal, target: BitwiseOperand, intermediate_dcc: u8) {
        {
            // if any row contains the signal, then this is easy, simply copy the row into the
            // target operand
            let signal_row = self.rows.get_rows(signal).next();
            if let Some(signal_row) = signal_row {
                self.set_operand_signal(target, signal);
                self.instructions
                    .push(Instruction::AAP(signal_row.into(), target.into()));
                return;
            }
        }
        // otherwise we need to search the inverted signal and take a DCC row if possible, otherwise
        // use the intermediate DCC to invert
        let mut inverted_signal_row = None;
        for row in self.rows.get_rows(signal.invert()) {
            if let Row::Bitwise(BitwiseRow::DCC(_)) = row {
                inverted_signal_row = Some(row);
                break;
            }
            inverted_signal_row = Some(row);
        }
        let inverted_signal_row =
            inverted_signal_row.expect("inverted signal row should be present");
        if let Row::Bitwise(BitwiseRow::DCC(dcc)) = inverted_signal_row {
            // alrighty, that's great, the inverted DCC row contains our signal
            // let's copy that over
            self.set_operand_signal(target, signal);
            self.instructions.push(Instruction::AAP(
                BitwiseOperand::DCC {
                    inverted: true,
                    index: dcc,
                }
                .into(),
                target.into(),
            ));
            return;
        }
        // this is the very sad case in which we have to use the intermediate DCC row to create the
        // signal from its inverse
        // first, copy the inverted signal from the signal_row to the DCC row
        let intermediate_dcc_operand = BitwiseOperand::DCC {
            inverted: false,
            index: intermediate_dcc,
        };
        self.set_operand_signal(intermediate_dcc_operand, signal.invert());
        self.instructions.push(Instruction::AAP(
            inverted_signal_row.into(),
            intermediate_dcc_operand.into(),
        ));

        // then copy the signal from the inverted intermediate DCC row into our target operand
        let inv_intermediate_dcc_operand = BitwiseOperand::DCC {
            inverted: true,
            index: intermediate_dcc,
        };
        self.set_operand_signal(target, signal);
        self.instructions.push(Instruction::AAP(
            inv_intermediate_dcc_operand.into(),
            target.into(),
        ));
    }

    /// Sets the value of the operand in `self.rows` to the given signal. If that removes the last
    /// reference to the node of the previous signal of the operator, insert spill code for the
    /// previous signal
    /// **ALWAYS** call this before inserting the actual instruction, otherwise the spill code will
    /// spill the wrong value
    fn set_operand_signal(&mut self, operand: BitwiseOperand, signal: Signal) {
        if let Some(previous_signal) = self.rows.set_operand_signal(operand, signal) {
            if !self.rows.contains_id(signal.node_id()) {
                let spill_id = self.rows.add_spill(previous_signal);
                self.instructions
                    .push(Instruction::AAP(operand.into(), Address::Spill(spill_id)));
            }
        }
    }

    pub fn rows(&self) -> &Rows {
        &self.rows
    }
}

impl SingleRowAddress {
    pub fn row(&self) -> Row {
        match self {
            SingleRowAddress::In(i) => Row::In(*i),
            SingleRowAddress::Spill(i) => Row::Spill(*i),
            SingleRowAddress::Const(i) => Row::Const(*i),
            SingleRowAddress::Bitwise(o) => o.row().into(),
        }
    }
    pub fn inverted(&self) -> bool {
        match self {
            SingleRowAddress::Bitwise(o) => o.inverted(),
            _ => false,
        }
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

impl From<BitwiseAddress> for Address {
    fn from(value: BitwiseAddress) -> Self {
        Self::Bitwise(value)
    }
}

impl From<BitwiseOperand> for BitwiseAddress {
    fn from(value: BitwiseOperand) -> Self {
        Self::Single(value)
    }
}

impl From<BitwiseOperand> for Address {
    fn from(value: BitwiseOperand) -> Self {
        Self::Bitwise(value.into())
    }
}

impl From<BitwiseOperand> for SingleRowAddress {
    fn from(value: BitwiseOperand) -> Self {
        Self::Bitwise(value)
    }
}

impl From<BitwiseRow> for BitwiseOperand {
    fn from(value: BitwiseRow) -> Self {
        match value {
            BitwiseRow::T(t) => BitwiseOperand::T(t),
            BitwiseRow::DCC(dcc) => BitwiseOperand::DCC {
                index: dcc,
                inverted: false,
            },
        }
    }
}

impl From<Row> for SingleRowAddress {
    fn from(value: Row) -> Self {
        match value {
            Row::In(i) => SingleRowAddress::In(i),
            Row::Spill(i) => SingleRowAddress::Spill(i),
            Row::Const(c) => SingleRowAddress::Const(c),
            Row::Bitwise(b) => SingleRowAddress::Bitwise(b.into()),
        }
    }
}

impl From<Row> for Address {
    fn from(value: Row) -> Self {
        Self::from(SingleRowAddress::from(value))
    }
}

impl From<SingleRowAddress> for Address {
    fn from(value: SingleRowAddress) -> Self {
        match value {
            SingleRowAddress::In(i) => Self::In(i),
            SingleRowAddress::Spill(i) => Self::Spill(i),
            SingleRowAddress::Const(c) => Self::Const(c),
            SingleRowAddress::Bitwise(o) => Self::Bitwise(BitwiseAddress::Single(o)),
        }
    }
}

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
                Address::Spill(i) => write!(f, "S{}", i),
                Address::Const(c) => write!(f, "C{}", if *c { "1" } else { "0" }),
                Address::Bitwise(b) => match b {
                    BitwiseAddress::Single(o) => write_operand(f, o),
                    BitwiseAddress::Maj(id) => {
                        let operands = self.architecture.maj_ops[*id];
                        write!(f, "[")?;
                        write_operand(f, &operands[0])?;
                        write!(f, ", ")?;
                        write_operand(f, &operands[1])?;
                        write!(f, ", ")?;
                        write_operand(f, &operands[2])?;
                        write!(f, "]")
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
