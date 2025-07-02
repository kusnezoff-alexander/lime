//! Computation of Compiling Costs

use eggmock::egg::{CostFunction, Id};
use eggmock::{AoigLanguage, egg::Language};
use log::debug;
use std::cmp::Ordering;
use std::ops;
use std::rc::Rc;

use super::architecture::{FCDRAMArchitecture, LogicOp};

pub struct CompilingCostFunction{}

// impl StackedPartialGraph { } // Do I need this??

/// TODO: add reliability as cost-metric
#[derive(Debug, Clone, Copy)]
pub struct CompilingCost {
    // partial: RefCell<Either<StackedPartialGraph, Rc<CollapsedPartialGraph>>>,
    /// Probability that the whole program will run successfully
    success_rate: f64,
    /// Estimation of program cost (from input logic-ops)
    program_cost: usize,
}

/// Needed to implement `enode.fold()` for computing overall cost from node together with its children
impl ops::Add<CompilingCost> for CompilingCost {
    type Output = CompilingCost;

    fn add(self, rhs: CompilingCost) -> Self::Output {
        CompilingCost {
            success_rate: self.success_rate * rhs.success_rate, // monotonically decreasing
            program_cost: self.program_cost + rhs.program_cost, // monotonically increasing
        }
    }
}

impl CostFunction<AoigLanguage> for CompilingCostFunction {
    type Cost = Rc<CompilingCost>;

    /// Compute cost of given `enode` using `cost_fn`
    ///
    /// Parameters determining cost of an enode:
    /// - distance of row-operands to sense amplifiers
    /// - operation:
    ///     - AND=<TODO>
    ///     - OR=<TODO>
    ///     - NOT=<TODO>
    ///
    /// TODO: NEXT
    /// - [ ] Subgraph direkt kompilieren ??
    fn cost<C>(&mut self, enode: &AoigLanguage, mut cost_fn: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // TODO: detect self-cycles, other cycles will be detected by compiling, which will result in an error
        // enode.children();
        // TODO: rewrite to `.fold()`

        // get op-cost of executing `enode`:
        let op_cost = match *enode {
            AoigLanguage::False | AoigLanguage::Input(_) => {
                CompilingCost {
                    success_rate: 1.0,
                    program_cost: 0,
                }
            },
            AoigLanguage::And([node1, node2]) => {

                let mem_cycles_and  = FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(LogicOp::AND)
                    .iter().fold(0, |acc, instr| { acc + instr.get_nr_memcycles() as usize });
                let expected_success_rate = 0.0; // TODO: do empirical analysis of success-rate matrices of ops and come up with good values
                CompilingCost {
                    success_rate: expected_success_rate,
                    program_cost: mem_cycles_and,
                }

            },
            AoigLanguage::Or([node1, node2]) => {
                let mem_cycles_or  = FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(LogicOp::OR)
                    .iter().fold(0, |acc, instr| { acc + instr.get_nr_memcycles() as usize });
                let expected_success_rate = 0.0; // TODO: do empirical analysis of success-rate matrices of ops and come up with good values
                CompilingCost {
                    success_rate: expected_success_rate,
                    program_cost: mem_cycles_or,
                }
            },
            // TODO: increase cost of NOT? (since it moves the value to another subarray!)
            // eg prefer `OR(a,b)` to `NOT(AND( NOT(a), NOT(b)))`
            AoigLanguage::Not(node) => {
                let mem_cycles_and  = FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(LogicOp::NOT)
                    .iter().fold(0, |acc, instr| { acc + instr.get_nr_memcycles() as usize });
                let expected_success_rate = 0.0; // TODO: do empirical analysis of success-rate matrices of ops and come up with good values

                CompilingCost {
                    success_rate: expected_success_rate,
                    program_cost: mem_cycles_and,
                }
            },
            _ => {
                // todo!();
                CompilingCost {
                    success_rate: 0.0,
                    program_cost: 7,
                }
                // 0 // TODO: implement for nary-ops, eg using `.children()`
            }
        };

        debug!("Folding {:?}", enode);
        Rc::new(enode.fold(op_cost, |sum, id| sum + *(cost_fn(id)) )) // TODO: doesn't work yet :/

        // Rc::new(CompilingCost {
        //     success_rate: 0.0,
        //     program_cost: 7,
        // })
    }
}

impl PartialEq for CompilingCost {
    fn eq(&self, other: &Self) -> bool {
        self.success_rate == other.success_rate && self.program_cost == other.program_cost
    }
}

/// First compare based on success-rate, then on program-cost
/// TODO: more fine-grained comparison !!
impl PartialOrd for CompilingCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.success_rate == other.success_rate {
            self.program_cost.partial_cmp(&other.program_cost)
        } else {
            self.success_rate.partial_cmp(&other.success_rate)
        }
    }
}


#[cfg(test)]
mod tests {
    use eggmock::egg::{self, rewrite, EGraph, Extractor, RecExpr, Rewrite, Runner};
    use eggmock::{AoigLanguage, ComputedNetworkWithBackwardEdges, Network, Signal};

    use crate::fc_dram::compiler::Compiler;
    use crate::fc_dram::egraph_extraction::CompilingCostFunction;
    use crate::fc_dram::CompilerSettings;

    use super::*;
    // import all elements from parent-module
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init() -> Compiler {
        INIT.call_once(|| {
            env_logger::init();
        });
        Compiler::new(CompilerSettings { print_program: true, verbose: true, print_compilation_stats: false, min_success_rate: 0.999, repetition_fracops: 5, safe_space_rows_per_subarray: 16 } )
    }

    /// TODO !
    #[test]
    fn test_cost_function () {
        let mut compiler = init();
        let rewrite_rules: Vec<Rewrite<AoigLanguage, ()>> =  vec![
                // TODO: add "or" - and De-Morgan ?
                rewrite!("commute-and"; "(and ?a ?b)" => "(and ?b ?a)"),
                rewrite!("and-1"; "(and ?a 1)" => "?a"),
                rewrite!("and-0"; "(and ?a 0)" => "0"),
                // TODO: first add `AOIG`-language and add conversion AOIG<->AIG (so mockturtle's aig can still be used underneath)
                rewrite!("and-or"; "(! (or (! ?a) (! ?b)))" => "(and ?a ?b)"), // (De-Morgan) ! not checked whether this works
                rewrite!("or-and"; "(! (and (! ?a) (! ?b)))" => "(or ?a ?b)" ), // (De-Morgan) ! not checked whether this works
                rewrite!("and-or-more-not"; "(and ?a ?b)" => "(! (or (! ?a) (! ?b)))"), // (De-Morgan) ! not checked whether this works
                rewrite!("or-and-more-not"; "(or ?a ?b)" => "(! (and (! ?a) (! ?b)))"), // (De-Morgan) ! not checked whether this works
                rewrite!("and-same"; "(and ?a ?a)" => "?a"),
                rewrite!("not_not"; "(! (! ?a))" => "?a"),
            ];

        let mut egraph: EGraph<AoigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AoigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
        egraph.add_expr(&my_expression);
        let output2 = egraph.add(AoigLanguage::And([eggmock::egg::Id::from(0), eggmock::egg::Id::from(2)])); // additional `And` with one src-operand=input and one non-input src operand
        debug!("EGraph used for candidate-init: {:?}", egraph);
        let egraph_clone = egraph.clone();
        let extractor = Extractor::new( &egraph_clone, CompilingCostFunction {});
        let ntk = &(extractor, vec!(egg::Id::from(5), output2));
        ntk.dump();
        // Id(5): And([Signal(false, Id(2)), Signal(false, Id(4))])
        // Id(4): And([Signal(false, Id(3)), Signal(false, Id(1))])
        // Id(1): Input(3)
        // Id(3): Input(2)
        // Id(2): And([Signal(false, Id(0)), Signal(false, Id(1))])
        // Id(0): Input(1)

        let runner = Runner::default().with_egraph(egraph).run(rewrite_rules.as_slice());

        let graph = runner.egraph;

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        // TODO: test-case with node that relies on one input src-operand and one non-input (intermediate node) src-operand
    }
}
