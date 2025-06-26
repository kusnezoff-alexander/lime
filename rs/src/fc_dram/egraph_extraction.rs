//! Computation of Compiling Costs

use eggmock::egg::{CostFunction, Id};
use eggmock::{EggIdToSignal, AoigLanguage, Aoig, NetworkLanguage, Network, Signal};
use std::cmp::Ordering;
use std::rc::Rc;

pub struct CompilingCostFunction{}

// impl StackedPartialGraph { } // Do I need this??

/// TODO: add reliability as cost-metric
#[derive(Debug)]
pub struct CompilingCost {
    // partial: RefCell<Either<StackedPartialGraph, Rc<CollapsedPartialGraph>>>,
    /// Probability that the whole program will run successfully
    success_rate: f64,
    /// Estimation of program cost (from input logic-ops)
    program_cost: usize,
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
        let cost = match enode {
            AoigLanguage::False => 0,
            AoigLanguage::Input(_node) => {
                // FCDRAMArchitecture::get_distance_of_row_to_sense_amps(&self, row)
                // TODO: make cost depend on data-pattern of input?
                0
            },
            AoigLanguage::And([_node1, _node2]) | AoigLanguage::Or([_node1, _node2]) => {
                // TODO: get mapping of AND to FCDRAM-Primitives and get how many mem-cycles they take
                3
            },
            // TODO: increase cost of NOT? (since it moves the value to another subarray!)
            // eg prefer `OR(a,b)` to `NOT(AND( NOT(a), NOT(b)))`
            AoigLanguage::Not(_node) => {
               100 // NOTs seem to be horrible (unless the computation proceeds in the other subarray where the NOT result is placed)
            },
            _ => {
                0 // TODO: implement for nary-ops
            }
        };

        Rc::new(CompilingCost {
            success_rate: 0.0, // TODO
            program_cost: cost,
        })
        // todo!()

        // let root = enode.clone();
        // let cost = match enode {
        //     AoigLanguage::False | AoigLanguage::Input(_) => CompilingCost::leaf(root),
        //     AoigLanguage::Not(id) => {
        //         let cost = costs(*id);
        //
        //         let nesting = if cost.not_nesting == NotNesting::NotANot {
        //             NotNesting::FirstNot
        //         } else {
        //             NotNesting::NestedNots
        //         };
        //         //
        //         CompilingCost::with_children(self.architecture, root, iter::once((*id, cost)), nesting)
        //     }
        //     AoigLanguage::Maj(children) => CompilingCost::with_children(
        //         self.architecture,
        //         root,
        //         children.map(|id| (id, costs(id))),
        //         NotNesting::NotANot,
        //     ),
        // };
        // Rc::new(cost)
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
