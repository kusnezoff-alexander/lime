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
    architecture::{FCDRAMArchitecture, Instruction, SubarrayId, ARCHITECTURE, NR_SUBARRAYS, ROWS_PER_SUBARRAY, SUBARRAY_ID_BITMASK}, optimization::optimize, CompilerSettings, Program, ProgramState, RowAddress
};
use eggmock::{Aig, AigLanguage, ComputedNetworkWithBackwardEdges, Id, NetworkLanguage, NetworkWithBackwardEdges, Node, Signal};
use log::debug;
use priority_queue::PriorityQueue;
use std::{cmp::Ordering, collections::{BinaryHeap, HashMap, HashSet}};

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

        while let Some((next_candidate, _)) = self.comp_state.candidates.pop() {
                // TODO: extend program with instr that is executed next
                let executed_instructions = &mut self.execute_next_instruction(&next_candidate, network);
                program.instructions.append(executed_instructions);

                // update new candidates
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
    /// - NOTE: nr of safe-space rows must be a power of 2 (x) between 1<=x<=64
    /// - [ ] TODO: improve algo (in terms of space efficiency)
    fn alloc_safe_space_rows(&mut self, nr_safe_space_rows: u8) {

        let supported_nr_safe_space_rows = vec!(1,2,4,8,16,32,64);
        if !supported_nr_safe_space_rows.contains(&nr_safe_space_rows) {
            panic!("Only the following nr of rows are supported to be activated: {:?}, given: {}", supported_nr_safe_space_rows, nr_safe_space_rows);
        }

        // TODO: this is just a quick&dirty implementation. Solving this (probably NP-complete) problem of finding optimal safe-space rows is probably worth solving for every DRAM-module once
        self.safe_space_rows = {

            // choose any row-addr combi activating exactly `nr_safe_space_rows` and choose all that activated rows to be safe-space rows
            let chosen_row_addr_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations
                .get(&nr_safe_space_rows).unwrap()
                .first().unwrap(); // just take the first row-combi that activates `nr_safe_space_rows`
            ARCHITECTURE.precomputed_simultaneous_row_activations.get(chosen_row_addr_combi).unwrap().to_vec()
        };

        // deactivate all combination which could activate safe-space rows
        for row in self.safe_space_rows.iter() {
            for row_combi in ARCHITECTURE.row_activated_by_rowaddress_tuple.get(row).unwrap() {
                self.blocked_row_combinations.insert(*row_combi);
            }
        }
    }

    /// Place inputs onto appropriate rows
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of move-ops to move intermediate results around & keep inputs close to sense-amps
    /// TODO: parititon logic network into subgraphs s.t. subgraphs can be mapped onto subarrays reducing nr of needed moves
    fn place_inputs(&mut self, mut inputs: Vec<Id>) {
        // naive implementation: start placing input on consecutive safe-space rows (continuing with next subarray once the current subarray has no more free safe-space rows)
        // TODO: change input operand placement to be more optimal (->taking into account future use of those operands)
        let mut subarray_iter=0..NR_SUBARRAYS;
        let next_subarray = subarray_iter.next().unwrap();
        let mut row_iter= self.safe_space_rows.iter();
        while let Some(next_input) = inputs.pop() {
            let (next_subarray, next_row) = if let Some(next_row) = row_iter.next() {
                (next_subarray, next_row)
            } else {
                row_iter = self.safe_space_rows.iter();
                ( subarray_iter.next().expect("OOM: no more safe-space rows and subarrays available") , row_iter.next().expect("No safe-space rows available" ))
            };


            let row_address = (next_subarray << ROWS_PER_SUBARRAY.ilog2() ) | next_row; // higher bits=id of subarray

            let initial_row_state = RowState { is_compute_row: false, live_value: Some(next_input)};
            let initial_value_state = ValueState { is_computed: true, row_location: Some(row_address)};
            self.comp_state.dram_state.insert(row_address, initial_row_state);
            self.comp_state.value_states.insert(next_input, initial_value_state);
        }
    }

    /// Initialize candidates with all nodes that are computable
    fn initialize_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aig>) {
        let inputs: Vec<Id> = network.leaves().collect();

        // init candidates with all nodes having only inputs as src-operands
        for &input in inputs.as_slice() {
            // TODO: NEXT - add all non-input nodes whose src-operands are all inputs
            let mut outputs_with_prio: PriorityQueue<Id, SchedulingPrio> = network.node_outputs(input)
                .filter(|output| network.node(*output).inputs().iter().all(|other_input| inputs.contains(&other_input.node_id()) )) // only those nodes are candidates, whose src-operands are ALL inputs (->only primary inputs are directly available)
                .map( |output| (output, self.compute_scheduling_prio_for_node(output, network)) )
                .collect();
            self.comp_state.candidates.append(&mut outputs_with_prio);
            debug!("{:?} has the following outputs: {:?}", input, network.node_outputs(input).collect::<Vec<Id>>());
        }
    }

    /// Initialize compilation state: mark unsuable rows (eg safe-space rows), place input operands
    fn init_comp_state(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aig>) {
        // debug!("Compiling {:?}", network);
        // 0.1 Allocate safe-space rows (for storing intermediate values safely
        self.alloc_safe_space_rows(self.settings.safe_space_rows_per_subarray);
        // 0.2 Place all inputs and mark them as being live
        self.place_inputs( network.leaves().collect::<Vec<Id>>() ); // place input-operands into rows
        debug!("Placed inputs {:?} in {:?}", network.leaves().collect::<Vec<Id>>(), self.comp_state.value_states);
        // 0.3 Setup: store all network-nodes yet to be compiled
        self.initialize_candidates(network);
    }

    /// Return id of subarray to use for computation and reference (compute_subarrayid, reference_subarrayid)
    /// - based on location of input rows AND current compilation state
    /// - [ ] POSSIBLE EXTENSION: include lookahead for future ops and their inputs they depend on
    fn select_compute_and_ref_subarray(&self, input_rows: Vec<RowAddress>) -> (SubarrayId, SubarrayId) {
        // naive implementation: just use the subarray that most of the `input_rows` reside in
        // TODO: find better solution
        let used_subarray_ids = input_rows.into_iter().map(|row| row & SUBARRAY_ID_BITMASK);
        let (&mostly_used_subarray_id, _) = used_subarray_ids
                .fold(HashMap::new(), |mut acc, item| {
                            *acc.entry(item).or_insert(0) += 1;
                            acc
                        })
                .iter().max_by_key(|&(_, count)| count).unwrap();

        let selected_ref_subarray = (mostly_used_subarray_id+1) % NR_SUBARRAYS; // TODO: use 2D-layout of subarrays to determine which of them share sense-amps

        (mostly_used_subarray_id, selected_ref_subarray)
    }

    /// - [ ] make sure that operation to be executed on those rows won't simultaneously activate other rows holding valid data which will be used by future operations
    fn execute_next_instruction(&mut self, next_candidate: &Id, network: &impl NetworkWithBackwardEdges<Node = Aig>) -> Vec<Instruction> {
        let src_operands: Vec<Id> = network.node(*next_candidate).inputs()
            .iter()
            .map(|signal| signal.node_id())
            .collect();

        let nr_operands = src_operands.len(); // use to select SRA to activate
        let nr_rows = nr_operands.next_power_of_two();

        let src_rows: Vec<RowAddress> = src_operands.iter()
            .map(|src_operand| {

                debug!("src: {src_operand:?}");
                self.comp_state.value_states.get(src_operand)
                    .unwrap()
                    .row_location
                    .expect("Sth went wrong... if the src-operand is not in a row, then this candidate shouldn't have been added to the list of candidates")
            })
            .collect();

        let (compute_subarray, ref_subarray) = self.select_compute_and_ref_subarray(src_rows);
        let language_op = network.node(*next_candidate);

        // match language_op {
        //     Aig::And()
        //             }

        // 0. Select an SRA (=row-address tuple) for the selected subarray based on highest success-rate
        // TODO (possible improvement): input replication by choosing SRA with more activated rows than operands and duplicating operands which are in far-away rows into several rows?)
        let (row1,row2) = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&(nr_rows as u8)).unwrap()
            // TODO: sort by success-rate
            .first().expect("No SRA for nr-rows={nr_rows}");

        // 1. Initialize rows in ref-subarray (if executing AND/OR)
        // - TODO: read nr of frac-ops to issue from compiler-settings

        // 2. Place rows in the simultaneously activated rows in the compute subarray (init other rows with 0 for OR, 1 for AND and same value for NOT)


        // SKIPPED: 2.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
        // - only necessary once we find optimization to not write values to safe-space but reuse them diectly
        // 2.2.1 if yes: move data to other rows for performing this op

        // 3. Issue actual operation

        // 4. Copy result data from dst NEAREST to the sense-amps into a safe-space row
        // TODO: possible improvement - error correction over all dst-rows (eg majority-vote for each bit, votes weighted by distance to sense-amps?)

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

#[derive(Debug)]
pub struct ValueState {
    /// Whether the value has already been computed (->only then it could reside in a row)
    /// - the value could also have been computed but spilled already on its last use
    /// - helps determining whether src-operand is the last use: for all other output operands of that source operand just check whether they have been already computed
    is_computed: bool,
    /// Row in which the value resides
    row_location: Option<RowAddress>,
}

/// Keep track of current progress of the compilation (eg which rows are used, into which rows data is placed, ...)
pub struct CompilationState {
    /// For each row in the dram-module store its state
    dram_state: HashMap<RowAddress, RowState>,
    /// Stores row in which an intermediate result (which is still to be used by future ops) is currently located (or whether it has been computed at all)
    value_states: HashMap<Id, ValueState>,
    /// List of candidates (ops ready to be issued) prioritized by some metric by which they are scheduled for execution
    /// - NOTE: calculate Nodes `SchedulingPrio` using
    candidates: PriorityQueue<Id, SchedulingPrio>,
}

impl CompilationState {
    pub fn new(dram_state: HashMap<RowAddress, RowState>) -> Self {
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

/// To execute the next instruction based on the following criteria:
/// 1. Select candidate which operates on most values which are used last (->to release safe-space right after)
/// 2. IF EQUAL: Select candidate with most successors (->pushes more candidates to select from in next step)
/// 3. IF EQUAL: Select any of the remaining candidates
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
    use std::sync::{LazyLock, Mutex, Once};

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

    static INIT: Once = Once::new();

    fn init() -> Compiler {
        INIT.call_once(|| {
            env_logger::init();
        });
        Compiler::new(CompilerSettings { print_program: true, verbose: true, print_compilation_stats: false, min_success_rate: 0.999, safe_space_rows_per_subarray: 16 } )
    }

    #[test] // mark function as test-fn
    fn test_alloc_safe_space_rows() {
        let mut compiler = init();
        const REQUESTED_SAFE_SPACE_ROWS: u8 = 8;
        compiler.alloc_safe_space_rows(REQUESTED_SAFE_SPACE_ROWS);

        assert_eq!(compiler.safe_space_rows.len(), REQUESTED_SAFE_SPACE_ROWS as usize);
    }

    #[test] // mark function as test-fn
    fn test_candidate_initialization() {
        let mut compiler = init();

        let mut egraph: EGraph<AigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
        egraph.add_expr(&my_expression);
        let output2 = egraph.add(AigLanguage::And([eggmock::egg::Id::from(0), eggmock::egg::Id::from(2)])); // additional `And` with one src-operand=input and one non-input src operand
        debug!("EGraph used for candidate-init: {:?}", egraph);
        let extractor = Extractor::new( &egraph, CompilingCostFunction {});
        let ntk = &(extractor, vec!(egg::Id::from(5), output2));
        ntk.dump();
        // Id(5): And([Signal(false, Id(2)), Signal(false, Id(4))])
        // Id(4): And([Signal(false, Id(3)), Signal(false, Id(1))])
        // Id(1): Input(3)
        // Id(3): Input(2)
        // Id(2): And([Signal(false, Id(0)), Signal(false, Id(1))])
        // Id(0): Input(1)

        // act is if one `AND` has already been computed -> other and (`Id(2)`) should be the only candidate left
        compiler .comp_state.value_states.insert(eggmock::Id::from(4), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        compiler.initialize_candidates(&ntk_backward);
        let is_candidate_ids: HashSet<Id> = compiler.comp_state.candidates.iter().map(|(id,_)| *id).collect();
        let should_candidate_ids: HashSet<Id> = HashSet::from([eggmock::Id::from(2), eggmock::Id::from(4)]);
        assert_eq!( is_candidate_ids, should_candidate_ids);

        // TODO: test-case with node that relies on one input src-operand and one non-input (intermediate node) src-operand
    }

    #[test]
    fn test_compute_scheduling_prio_for_node() {
        let mut compiler = init();

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
        compiler.comp_state.value_states.insert(eggmock::Id::from(2), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        let scheduling_prio = compiler.compute_scheduling_prio_for_node(eggmock::Id::from(4),  &ntk_backward);
        assert_eq!(scheduling_prio, SchedulingPrio { nr_last_value_uses: 2, nr_src_operands: 2, nr_result_operands: 1 }  );

    }

    #[test]
    fn test_select_compute_and_ref_subarray() {
        let compiler = init();
        let (selected_subarray, _) = compiler.select_compute_and_ref_subarray(vec!(0b1_000_000_000, 0b1_000_010_000, 0b111_000_000_000, 0b10_100_000_000,));
        assert_eq!(selected_subarray, 0b1_000_000_000);
    }
}
