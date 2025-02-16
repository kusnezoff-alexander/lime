mod compilation;
mod extraction;
mod optimization;
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
    maj_ops: Vec<usize>,
    multi_activations: Vec<Vec<BitwiseOperand>>,
    num_dcc: u8,
}

impl Architecture {
    pub fn new(multi_activations: Vec<Vec<BitwiseOperand>>, num_dcc: u8) -> Self {
        let maj_ops = multi_activations
            .iter()
            .enumerate()
            .filter(|(_, ops)| ops.len() == 3)
            .map(|(i, _)| i)
            .collect();
        Self {
            maj_ops,
            multi_activations,
            num_dcc,
        }
    }
}

impl BitwiseOperand {
    pub fn row(&self) -> BitwiseRow {
        match self {
            BitwiseOperand::T(t) => BitwiseRow::T(*t),
            BitwiseOperand::DCC { index, .. } => BitwiseRow::DCC(*index),
        }
    }
    pub fn is_dcc(&self) -> bool {
        matches!(self, BitwiseOperand::DCC { .. })
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
        let architecture = Architecture::new(
            vec![
                vec![T(0), T(1), T(2)],
                vec![T(1), T(2), T(3)],
                vec![
                    DCC {
                        index: 0,
                        inverted: false,
                    },
                    T(1),
                    T(2),
                ],
                vec![
                    DCC {
                        index: 0,
                        inverted: false,
                    },
                    T(0),
                    T(3),
                ],
            ],
            2,
        );

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
