mod compilation;
mod extraction;
mod program;
mod rows;

use self::compilation::compile;
use self::extraction::CompilingCostFunction;

use eggmock::egg::{rewrite, EGraph, Extractor, Id, Runner};
use eggmock::{
    Mig, MigLanguage, MigNode, MigReceiverFFI, Provider, Receiver, Rewriter, RewriterFFI,
};
use program::*;
use rows::*;
use std::time::Duration;

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

pub struct AmbitRewriter {}

impl Rewriter for AmbitRewriter {
    type Network = Mig;
    type Intermediate = (EGraph<MigLanguage, ()>, Vec<Id>);
    type Receiver = EGraph<MigLanguage, ()>;

    fn create_receiver(&mut self) -> Self::Receiver {
        EGraph::new(())
    }

    fn rewrite(
        self,
        (graph, outputs): (EGraph<MigLanguage, ()>, Vec<Id>),
        output: impl Receiver<Node = MigNode, Result = ()>,
    ) {
        use BitwiseOperand::*;
        let architecture = Architecture {
            maj_ops: vec![
                [T(0), T(1), T(2)],
                [T(1), T(2), T(3)],
                [
                    DCC {
                        index: 0,
                        inverted: false,
                    },
                    T(1),
                    T(2),
                ],
                [
                    DCC {
                        index: 0,
                        inverted: false,
                    },
                    T(0),
                    T(3),
                ],
            ],
            num_dcc: 2,
        };

        let rules = &[
            rewrite!("commute_1"; "(maj ?a ?b ?c)" => "(maj ?b ?a ?c)"),
            rewrite!("commute_2"; "(maj ?a ?b ?c)" => "(maj ?a ?c ?b)"),
            rewrite!("example"; "(maj (! f) ?a (maj f ?b ?c))" => "(maj (! f) ?a (maj ?a ?b ?c))"),
        ];
        let runner = Runner::default()
            .with_time_limit(Duration::from_secs(60))
            .with_egraph(graph)
            .run(rules);
        runner.print_report();
        let graph = runner.egraph;
        graph
            .dot()
            .to_dot("egraph.dot")
            .expect("dotting should not fail");

        let extractor = Extractor::new(
            &graph,
            CompilingCostFunction {
                architecture: &architecture,
            },
        );
        let ntk = (extractor, outputs);
        let ntk = ntk.with_backward_edges();
        let program = compile(&architecture, &ntk);
        println!("{program}");
        ntk.send(output);
    }
}

#[no_mangle]
extern "C" fn ambit_rewriter() -> MigReceiverFFI<RewriterFFI<Mig>> {
    RewriterFFI::new(AmbitRewriter {})
}
