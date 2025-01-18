mod compilation;
mod program;
mod rows;

pub use compilation::compile;
use eggmock::egg::{CostFunction, EGraph, Extractor, Id, Language};
use eggmock::{MigLanguage, MigReceiverFFI, Provider, Receiver, ReceiverFFI};
use program::*;
use rows::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitwiseOperand {
    T(u8),
    DCC { inverted: bool, index: u8 },
}

#[derive(Clone, Debug)]
pub struct Architecture {
    maj_ops: Vec<[BitwiseOperand; 3]>,
    num_dcc: u8,
}

impl BitwiseOperand {
    pub fn row(&self) -> BitwiseRow {
        match self {
            BitwiseOperand::T(t) => BitwiseRow::T(*t),
            BitwiseOperand::DCC { index, .. } => BitwiseRow::DCC(*index),
        }
    }
    pub fn inverted(&self) -> bool {
        matches!(self, BitwiseOperand::DCC { inverted: true, .. })
    }
}

#[no_mangle]
extern "C" fn ambit_compiler() -> MigReceiverFFI<()> {
    MigReceiverFFI::new(EGraph::<MigLanguage, ()>::new(()).map(|(graph, outputs)| {
        use BitwiseOperand::*;
        let architecture = Architecture {
            maj_ops: vec![
                [T(0), T(1), T(2)],
                [T(1), T(2), T(3)],
                [DCC {
                    index: 0,
                    inverted: false
                }, T(1), T(2)],
                [DCC {
                    index: 0,
                    inverted: false
                }, T(0), T(3)]
            ],
            num_dcc: 2,
        };
        graph.dot().to_dot("egraph.dot").expect("dotting should not fail");

        let extractor = Extractor::new(&graph, ConstCostFunction);
        let ntk = (extractor, outputs);
        let ntk = ntk.with_backward_edges();
        let program = compile(&architecture, &ntk);
        println!("{program}")
    }))
}

pub struct ConstCostFunction;

impl<L: Language> CostFunction<L> for ConstCostFunction {
    type Cost = u32;

    fn cost<C>(&mut self, _: &L, _: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        1
    }
}
