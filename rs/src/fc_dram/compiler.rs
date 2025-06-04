//!
//! - [`Compiler`] = performs actual compilation
//!     - [`CompilationState`] = stores states encountered during compilation (eg which values reside in which rows, rows containing live values, ..)
//!         - also see [`RowState`]
//! - [`SchedulingPrio`] = used to prioritize/order instruction for Instruction Scheduling
//!
//! - [`compile()`] = main function - compiles given logic network for the given [`architecture`]
//! into a [`program`] using some [`optimization`]

use crate::ambit::Architecture;

use super::{
    architecture::{FCDRAMArchitecture, Instruction, SubarrayId, ARCHITECTURE, NR_SUBARRAYS, ROWS_PER_SUBARRAY}, optimization::optimize, CompilerSettings, Program, ProgramState, RowAddress
};
use eggmock::{Aig, AigLanguage, ComputedNetworkWithBackwardEdges, Id, NetworkWithBackwardEdges, Node, Signal};
use log::debug;
use priority_queue::PriorityQueue;
use std::{cmp::Ordering, collections::{BinaryHeap, HashMap, HashSet}};


pub fn place_signals_onto_rows(
    comp_state: &mut impl NetworkWithBackwardEdges<Node = Aig>, signals: Vec<Id>
) {

}

pub struct Compiler {
    /// compiler-options set by user
    settings: CompilerSettings,
    /// Stores the state of all rows at each compilation step
    comp_state: CompilationState,
    /// These rows are reserved in EVERY subarray for storing intermediate results
    /// - NOTE: only initialized when compilation starts
    safe_space_rows: Vec<RowAddress>,
    /// RowCombinations which are not allowed to be issued via `APA` since they activate rows within the safe-space
    blocked_row_combinations: HashSet<(RowAddress,RowAddress)>,
}

impl Compiler {
    pub fn new(settings: CompilerSettings) -> Self {
        Compiler{
           settings,
           comp_state: CompilationState::new( HashMap::new() ),
           safe_space_rows: vec!(),
           blocked_row_combinations: HashSet::new(),
        }
    }

    /// Compiles given `network` into a FCDRAM-[`Program`] that can be run on given `architecture`
    ///
    /// General Procedure of compilation
    /// 1) Map Logical-Ops to FCDRAM-Primitives (operating on virtual rows)
    /// 2) Map virtual rows to actual physical rows (spilling/moving rows if necessary using `RowClone`)
    ///     - similarly to Register Allocation
    ///
    /// - [ ]  TODO: increase success-rate using input replication ? at which point to add input replication?
    ///
    /// - [ ] TODO: output in which rows
    ///     - 1) data is expected to be placed before program runs
    ///     - 2) outputs can be found after the program has run
    pub fn compile(
        &mut self,
        network: &impl NetworkWithBackwardEdges<Node = Aig>,
    ) -> Program {

        // 0. Prepare compilation: select safe-space rows, place inputs into DRAM module
        self.init_comp_state(network);

        let mut program = Program::new(vec!());

        let outputs = network.outputs();


        // TODO: how to get src-operands of `outputs` ??
        // debug!("{:?}", network.node_outputs(outputs.next()).collect());
        // TODO: get src-operands of outputs and place them appropriately (with knowledge about output
        // operands!)

        // start with inputs
        let primary_inputs = network.leaves();
        debug!("Primary inputs: {:?}", primary_inputs.collect::<Vec<Id>>());

        // println!("{:?}", network.outputs().collect::<Vec<Signal>>());
        debug!("Nodes in network:");
        for node in network.iter() {
            debug!("{:?},", node);
        }

        while !self.comp_state.candidates.is_empty() {
            // TODO: extend program with instr that is executed next
            let executed_instructions = &mut self.execute_next_instruction();
            program.instructions.append(executed_instructions);
        }

        // let (outputs, leaves) = (network.outputs(), network.leaves());

        // Program {
        //     instructions: vec!(Instruction::FracOp(-1)) ,
        // }
        // todo!()
        // optimize(&mut program);
        program
    }

    /// Allocates safe-space rows inside the DRAM-module
    /// - [ ] TODO: improve algo (in terms of space efficiency)
    fn alloc_safe_space_rows(&mut self, nr_safe_space_rows: u8) {

        let supported_nr_safe_space_rows = vec!(1,2,4,8,16,32,64);
        if !supported_nr_safe_space_rows.contains(&nr_safe_space_rows) {
            panic!("Only the following nr of rows are supported to be activated: {:?}, given: {}", supported_nr_safe_space_rows, nr_safe_space_rows);
        }

        // TODO: this is just a quick&dirty implementation. Solving this (probably NP-complete) problem of finding optimal safe-space rows is probably worth solving for every DRAM-module once
        let safe_space_rows = {

            // choose any row-addr combi activating exactly `nr_safe_space_rows` and choose all that activated rows to be safe-space rows
            let chosen_row_addr_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations
                .get(&nr_safe_space_rows).unwrap()
                .first().unwrap(); // just take the first row-combi that activates `nr_safe_space_rows`
            ARCHITECTURE.precomputed_simultaneous_row_activations.get(chosen_row_addr_combi).unwrap()
        };

        // deactivate all combination which could activate safe-space rows
        for row in safe_space_rows {
            for row_combi in ARCHITECTURE.row_activated_by_rowaddress_tuple.get(row).unwrap() {
                self.blocked_row_combinations.insert(*row_combi);
            }
        }
    }

    /// Place inputs onto appropriate rows
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of move-ops to move intermediate results around & keep inputs close to sense-amps
    /// TODO: parititon logic network into subgraphs s.t. subgraphs can be mapped onto subarrays reducing nr of needed moves
    fn place_inputs(&mut self, mut inputs: Vec<Id>) {
        // naive implementation: start placing input just on consecutive rows
        // TODO: change input operand placement to be more optimal (->taking into account future use of those operands)
        for subarray in 0..NR_SUBARRAYS {
            for row in self.safe_space_rows.iter() {
                // TODO  allocate safe-space rows
                if let Some(next_input) = inputs.pop() {
                    let initial_row_state = RowState { is_compute_row: false, live_value: Some(next_input)};
                    let initial_value_state = ValueState { is_computed: true, row_location: Some((subarray, *row))};
                    self.comp_state.dram_state.insert((subarray, *row), initial_row_state);
                    self.comp_state.value_states.insert(next_input, initial_value_state);
                } else {
                    return;
                }
            }
        }
    }

    /// Initialize candidates with all nodes that are computable
    fn initialize_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aig>) {
        let inputs = network.leaves();

        // init candidates with all nodes having inputs as src-operands

        for input in inputs {
            // TODO: NEXT - add all non-input nodes whose src-operands are all inputs
            let mut outputs_with_prio: PriorityQueue<Id, SchedulingPrio> = network.node_outputs(input)
                .map( |output| (output, self.compute_scheduling_prio_for_node(output, network)) )
                .collect();
            self.comp_state.candidates.append(&mut outputs_with_prio);
            debug!("{:?} has the following outputs: {:?}", input, network.node_outputs(input).collect::<Vec<Id>>());
        }

        debug!("Selected candidates: {:?}", self.comp_state.candidates);
        // delete candidates which have non-input src operands
    }

    /// Initialize compilation state: mark unsuable rows (eg safe-space rows), place input operands
    fn init_comp_state(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aig>) {
        // debug!("Compiling {:?}", network);
        // 0.1 Allocate safe-space rows (for storing intermediate values safely
        self.alloc_safe_space_rows(self.settings.safe_space_rows_per_subarray);
        // 0.2 Place all inputs and mark them as being live
        self.place_inputs( network.leaves().collect::<Vec<Id>>() ); // place input-operands into rows
        // 0.3 Setup: store all network-nodes yet to be compiled
        self.initialize_candidates(network);
    }

    /// Executes the next instruction based on the following criteria:
    /// 1. Select candidate which operates on most values which are used last (->to release safe-space right after)
    /// 2. IF EQUAL: Select candidate with most successors (->pushes more candidates to select from in next step)
    /// 3. IF EQUAL: Select any of the remaining candidates
    ///
    /// - [ ] make sure that operation to be executed on those rows won't simultaneously activate other rows holding valid data which will be used by future operations
    fn execute_next_instruction(&self) -> Vec<Instruction> {
        // 1. Make sure rows are placed appropriately in the rows to be simultaneously activated (starting from inputs)

        // 1.1 Determine in which rows src-operands for the next candidate-op are located

        // 1.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
        // 1.2.1 if yes: move data to other rows for performing this op

        // 1.3 Prepare performing the actual op (setup reference subarray)
        // 1.3.1 If activated rows in reference subarray holds valid data: spill to other rows

        // 1.4 Issue actual operation
        vec!()
    }

    /// Compute `SchedulingPrio` for a given node
    /// - used for inserting new candidates
    /// TODO: write unittest for this function
    fn compute_scheduling_prio_for_node(&self, id: Id, network: &impl NetworkWithBackwardEdges<Node = Aig>) -> SchedulingPrio {
        let nr_last_value_uses = network.node(id).inputs() // for each input check whether `id` is the last node using it
            .iter()
            .fold(0, |acc, input| {
                let input_id = Signal::node_id(input);
                let non_computed_outputs: Vec<Id> = network.node_outputs(input_id) // get all other nodes relying on this input
                    .filter(|out|
                        *out != id &&
                        !(self.comp_state.value_states.get(out)
                            .unwrap_or(&ValueState{is_computed: false, row_location: None}) // no entry means this is the first time accessing this value
                            .is_computed)
                    ) // filter for uses of `input` which still rely on it (=those that are not computed yet, except for currently checked node
                    .collect();
                if non_computed_outputs.is_empty() {
                    acc + 1
                } else {
                    acc
                }
            });

        SchedulingPrio {
            nr_last_value_uses,
            nr_src_operands: network.node(id).inputs().len(),
            nr_result_operands:  network.node_outputs(id).collect::<Vec<Id>>().len(),
        }
    }
}

/// Stores the current state of a row at a concrete compilations step
pub struct RowState {
    /// True iff that row is currently: 1) Not a safe-sapce row, 2) Doesn't activate any safe-sapce rows, 3) Isn't holding valid values in the role of a reference-subarray row
    is_compute_row: bool,
    /// `None` if the value inside this row is currently not live
    live_value: Option<Id>,
}

pub struct ValueState {
    /// Whether the value has already been computed (->only then it could reside in a row)
    /// - the value could also have been computed but spilled already on its last use
    /// - helps determining whether src-operand is the last use: for all other output operands of that source operand just check whether they have been already computed
    is_computed: bool,
    /// Row in which the value resides
    row_location: Option<(SubarrayId, RowAddress)>,
}

/// Keep track of current progress of the compilation (eg which rows are used, into which rows data is placed, ...)
pub struct CompilationState {
    /// For each row in the dram-module store its state
    dram_state: HashMap<(SubarrayId, RowAddress), RowState>,
    /// Stores row in which an intermediate result (which is still to be used by future ops) is currently located (or whether it has been computed at all)
    value_states: HashMap<Id, ValueState>,
    /// List of candidates (ops ready to be issued) prioritized by some metric by which they are scheduled for execution
    /// - NOTE: calculate Nodes `SchedulingPrio` using
    candidates: PriorityQueue<Id, SchedulingPrio>,
}

impl CompilationState {
    pub fn new(dram_state: HashMap<(SubarrayId, RowAddress), RowState>) -> Self {
        Self {
            dram_state,
            value_states: HashMap::new(),
            candidates: PriorityQueue::new(),
        }
    }
}

// impl<'n, P: NetworkWithBackwardEdges<Node = Aig>> CompilationState<'n, P> {
//     /// initializes `self..candidates()` with inputs + nodes whose src-operands are all inputs
//     pub fn new(network: &'n P) -> Self {
//         let mut candidates = FxHashSet::default();
//         // check all parents of leaves whether they have only leaf children, in which case they are
//         // candidates (since all of their inputs are calculated then)
//         for leaf in network.leaves() { // =inputs
//             for candidate_id in network.node_outputs(leaf) {
//                 let candidate = network.node(candidate_id);
//                 if candidate
//                     .inputs()
//                         .iter()
//                         .all(|signal| network.node(signal.node_id()).is_leaf())
//                 {
//                     candidates.insert((candidate_id, candidate));
//                 }
//             }
//         }
//
//         // let outputs = network
//         //     .outputs()
//         //     .enumerate()
//         //     .map(|(id, sig)| (sig.node_id(), (id as i64, sig)))
//         //     .collect();
//
//         let total_nr_rows_in_dram_module = ARCHITECTURE.nr_subarrays;
//         Self {
//             network,
//             candidates,
//             signal_to_row_mapping: HashMap::new(),
//             free_rows: (0..=total_nr_rows_in_dram_module).collect(), // we start with all rows being free at the beginning
//         }
//     }
// }

/// Contains info to order nodes for Instruction Scheduling
/// GOAL: minimize register usage:
///     1. Number of last-value-uses
///     2. Number src operands: ASSUMPTIONS=executing that op reduces value-usage of all of those inputs, higher prob. of last-use in next steps)
///     3. Total number of nodes that have `node` as input-operands
///         - possible extension: weigh result-operands based on how many of their src-operands are already computed
#[derive(PartialEq, Eq, Debug)]
struct SchedulingPrio {
    /// Nr of values which are used last by that node
    nr_last_value_uses: u64,
    /// Number of source operands the Node has
    nr_src_operands: usize,
    /// Number of result operands the Node has
    nr_result_operands: usize,
}

impl Ord for SchedulingPrio {
    fn cmp(&self, other: &Self) -> Ordering {
        self.nr_last_value_uses.cmp(&other.nr_last_value_uses)
            .then(self.nr_src_operands.cmp(&other.nr_src_operands)) // if `nr_last_value_uses` is equal
            .then(self.nr_result_operands.cmp(&other.nr_result_operands))
    }
}

impl PartialOrd for SchedulingPrio {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use eggmock::egg::{self, EGraph, Extractor, RecExpr};
    use eggmock::Network;

    use crate::fc_dram::egraph_extraction::CompilingCostFunction;

    use super::*; // import all elements from parent-module
    use std::sync::{LazyLock, Mutex};

    // ERROR: `eggmock`-API doesn't allow this..
    // // For data shared among unittests but initalized only once
    // static TEST_DATA: LazyLock<_> = LazyLock::new(|| {
    //     let mut egraph: EGraph<AigLanguage, ()> = Default::default();
    //     let my_expression: RecExpr<AigLanguage> = "(and (and a c) (and b c))".parse().unwrap();
    //     let extractor = Extractor::new( &egraph, CompilingCostFunction {});
    //     let ntk = (extractor, vec!(egg::Id::from(9)));
    //
    //     ComputedNetworkWithBackwardEdges::new(&ntk)
    // });

    static TEST_COMPILER: LazyLock<Mutex<Compiler>> = LazyLock::new(|| {
        Mutex::new(
            Compiler::new(CompilerSettings { print_program: true, verbose: true, print_compilation_stats: false, min_success_rate: 0.999, safe_space_rows_per_subarray: 16 } )
        )
    });

    #[test] // mark function as test-fn
    fn test_candidate_initialization() {
        env_logger::init();

        let mut egraph: EGraph<AigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
        egraph.add_expr(&my_expression);
        let extractor = Extractor::new( &egraph, CompilingCostFunction {});
        let ntk = &(extractor, vec!(egg::Id::from(5)));
        ntk.dump();
        // Id(5): And([Signal(false, Id(2)), Signal(false, Id(4))])
        // Id(4): And([Signal(false, Id(3)), Signal(false, Id(1))])
        // Id(1): Input(3)
        // Id(3): Input(2)
        // Id(2): And([Signal(false, Id(0)), Signal(false, Id(1))])
        // Id(0): Input(1)

        // act is if one `AND` has already been computed -> other and (`Id(2)`) should be the only candidate left
        TEST_COMPILER.lock().unwrap()
            .comp_state.value_states.insert(eggmock::Id::from(4), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        TEST_COMPILER.lock().unwrap().initialize_candidates(&ntk_backward);
        println!("{:?}", TEST_COMPILER.lock().unwrap().comp_state.candidates);
        // let result = add(2, 2);
        // assert_eq!(result, 4, "Result isn't equal to 4 :/");
    }

    #[test]
    fn test_compute_scheduling_prio_for_node() {
        env_logger::init();

        let mut egraph: EGraph<AigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
        egraph.add_expr(&my_expression);
        let extractor = Extractor::new( &egraph, CompilingCostFunction {});
        let ntk = &(extractor, vec!(egg::Id::from(5)));
        ntk.dump();
        // Id(5): And([Signal(false, Id(2)), Signal(false, Id(4))])
        // Id(4): And([Signal(false, Id(3)), Signal(false, Id(1))])
        // Id(1): Input(3)
        // Id(3): Input(2)
        // Id(2): And([Signal(false, Id(0)), Signal(false, Id(1))])
        // Id(0): Input(1)

        // act is if one `AND` has already been computed -> other and (`Id(2)`) should be the only candidate left
        TEST_COMPILER.lock().unwrap()
            .comp_state.value_states.insert(eggmock::Id::from(2), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        let scheduling_prio = TEST_COMPILER.lock().unwrap().compute_scheduling_prio_for_node(eggmock::Id::from(4),  &ntk_backward);
        assert_eq!(scheduling_prio, SchedulingPrio { nr_last_value_uses: 2, nr_src_operands: 2, nr_result_operands: 1 }  );

    }
}
