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
    architecture::{subarrayid_to_subarray_address, FCDRAMArchitecture, Instruction, LogicOp, SubarrayId, ARCHITECTURE, NR_SUBARRAYS, ROWS_PER_SUBARRAY, SUBARRAY_ID_BITMASK}, optimization::optimize, CompilerSettings, Program, ProgramState, RowAddress
};
use eggmock::{Aig, AigLanguage, ComputedNetworkWithBackwardEdges, Id, NetworkLanguage, NetworkWithBackwardEdges, Node, Signal};
use itertools::Itertools;
use log::debug;
use priority_queue::PriorityQueue;
use rustc_hash::FxHashMap;
use std::{cmp::Ordering, collections::{BinaryHeap, HashMap, HashSet}, env::consts::ARCH, fs::write};

pub struct Compiler {
    /// compiler-options set by user
    settings: CompilerSettings,
    /// Stores the state of all rows at each compilation step
    comp_state: CompilationState,
    /// These rows are reserved in EVERY subarray for storing intermediate results (ignore higher bits of these RowAddress)
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

        debug!("[ENTER alloc_safe_space_rows]");
        let supported_nr_safe_space_rows = vec!(1,2,4,8,16,32,64);
        if !supported_nr_safe_space_rows.contains(&nr_safe_space_rows) {
            panic!("Only the following nr of rows are supported to be activated: {:?}, given: {}", supported_nr_safe_space_rows, nr_safe_space_rows);
        }

        // TODO: this is just a quick&dirty implementation. Solving this (probably NP-complete) problem of finding optimal safe-space rows is probably worth solving for every DRAM-module once
        self.safe_space_rows = {

            // choose any row-addr combi activating exactly `nr_safe_space_rows` and choose all that activated rows to be safe-space rows
            let chosen_row_addr_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations
                .get(&nr_safe_space_rows).unwrap()
                .iter().next().unwrap(); // just take the first row-combi that activates `nr_safe_space_rows`
            ARCHITECTURE.precomputed_simultaneous_row_activations.get(chosen_row_addr_combi).unwrap().to_vec()
        };

        // deactivate all combination which could activate safe-space rows
        for row in self.safe_space_rows.iter() {
            for row_combi in ARCHITECTURE.row_activated_by_rowaddress_tuple.get(row).unwrap() {
                self.blocked_row_combinations.insert(*row_combi);
            }
        }
    }

    /// Places (commonly used) constants in safe-space rows
    /// - ! all safe-space rows are assumed to be empty when placing constants (constans are the first things to be placed into safe-space rows)
    /// - currently placed constants: all 0s and all 1s (for [`Compiler::init_reference_subarray`]
    fn place_constants(&mut self) {
        // place constants in EVERY subarray
        for subarray in 0..NR_SUBARRAYS {
            let mut safe_space = self.safe_space_rows.iter();
            let row_address_0 = subarrayid_to_subarray_address(subarray) | safe_space.next().unwrap();
            self.comp_state.constant_values.insert(0, row_address_0);
            let row_address_1 = subarrayid_to_subarray_address(subarray) | safe_space.next().unwrap();
            self.comp_state.constant_values.insert(1, row_address_1);

            self.comp_state.dram_state.insert(row_address_0, RowState { is_compute_row: false, live_value: None, constant: Some(0)} );
            self.comp_state.dram_state.insert(row_address_1, RowState { is_compute_row: false, live_value: None, constant: Some(1)} );
        }
    }

    /// Place inputs onto appropriate rows
    /// - NOTE: constants are expected to be placed before the inputs
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of move-ops to move intermediate results around & keep inputs close to sense-amps
    /// TODO: parititon logic network into subgraphs s.t. subgraphs can be mapped onto subarrays reducing nr of needed moves
    fn place_inputs(&mut self, mut inputs: Vec<Id>) {
        debug!("[ENTER place_inputs]");
        // naive implementation: start placing input on consecutive safe-space rows (continuing with next subarray once the current subarray has no more free safe-space rows)
        // TODO: change input operand placement to be more optimal (->taking into account future use of those operands)
        let mut subarray_iter=0..NR_SUBARRAYS;
        let next_subarray = subarray_iter.next().unwrap();
        let subarray_addr = subarrayid_to_subarray_address(next_subarray);
        while let Some(next_input) = inputs.pop() {
            let mut row_iter= self.safe_space_rows.iter()
                .filter(|row| ! self.comp_state.constant_values.values().contains( &(subarray_addr | **row)) ); // filter safe-space rows which are NOT already used for constants
            // NOTE: some safe-space rows are reserved for constants
            let (next_subarray, next_row) = if let Some(next_row) = row_iter.next() {
                (next_subarray, next_row)
            } else {
                let next_subarray = subarray_iter.next().expect("OOM: no more safe-space rows and subarrays available");
                let next_subarray_addr = subarrayid_to_subarray_address(next_subarray);
                let mut row_iter = self.safe_space_rows.iter()
                    .filter(|row| ! self.comp_state.constant_values.values().contains(&(next_subarray_addr | **row )) ); // filter safe-space rows which are NOT already used for constants
                ( next_subarray, row_iter.next().expect("OOM: No safe-space rows available" ) )
            };


            let row_address = (next_subarray << ROWS_PER_SUBARRAY.ilog2() ) | next_row; // higher bits=id of subarray

            let initial_row_state = RowState { is_compute_row: false, live_value: Some(next_input), constant: None };
            let initial_value_state = ValueState { is_computed: true, row_location: Some(row_address) };
            self.comp_state.dram_state.insert(row_address, initial_row_state);
            self.comp_state.value_states.insert(next_input, initial_value_state);
        }
    }

    /// Initialize candidates with all nodes that are computable
    fn initialize_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aig>) {
        debug!("[ENTER initialize_candidates]");
        let inputs: Vec<Id> = network.leaves().collect();

        // init candidates with all nodes having only inputs as src-operands
        for &input in inputs.as_slice() {
            // every output has a prio determined eg by how many src-operands it uses last (->to minimize nr of live values in rows)
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

        self.place_constants();
        // 0.2 Place all inputs and mark them as being live
        self.place_inputs( network.leaves().collect::<Vec<Id>>() ); // place input-operands into rows
        debug!("Placed inputs {:?} in {:?}", network.leaves().collect::<Vec<Id>>(), self.comp_state.value_states);
        // 0.3 Setup: store all network-nodes yet to be compiled
        self.initialize_candidates(network);
    }

    /// Returns instructions to initialize `ref_rows` in reference-subarray for corresponding logic-op
    /// - NOTE: [1] doesn't describe how the 0s/1s get into the reference subarray. We use `RowCloneFPM` ([4])) to copy the constant 0s/1s from the reserved safe-space row into the corresponding reference subarray row
    fn init_reference_subarray(&self, mut ref_rows: Vec<RowAddress>, logic_op: LogicOp) -> Vec<Instruction> {
        match logic_op {
            LogicOp::AND => {
                let frac_row = ref_rows.pop().expect("Min 1 row has to be passed for initializing ref subarray"); // TODO: include success-rate considerations to choose best row to use for storing `V_{DD}/2`
                let row_address_1 = self.comp_state.constant_values.get(&1).expect("Constants are expected to be placed in every subarray beforehand"); // row address where all 1s (V_DD) is stored
                let mut instructions = vec!();
                for _ in 0..self.settings.repetition_fracops {
                    instructions.push(Instruction::FracOp(frac_row));
                }
                for other_row in ref_rows {
                    instructions.push(Instruction::RowCloneFPM(*row_address_1, other_row));
                }
                instructions
            },
            LogicOp::OR => {
                let frac_row = ref_rows.pop().expect("Min 1 row has to be passed for initializing ref subarray"); // TODO: include success-rate considerations to choose best row to use for storing `V_{DD}/2`
                let row_address_0 = self.comp_state.constant_values.get(&0).expect("Constants are expected to be placed in every subarray beforehand"); // row address where all 1s (V_DD) is stored
                let mut instructions = vec!();
                for _ in 0..self.settings.repetition_fracops {
                    instructions.push(Instruction::FracOp(frac_row));
                }
                for other_row in ref_rows {
                    instructions.push(Instruction::RowCloneFPM(*row_address_0, other_row));
                }
                instructions
            },
            LogicOp::NOT => vec!(),
            _ => panic!("{logic_op:?} not supported yet"),
        }
    }

    /// Places the referenced `src_operands` into the corresponding `row_addresses` which are expected to be simultaneously executed
    fn init_compute_subarray(&mut self, mut row_addresses: Vec<RowAddress>, mut src_operands: Vec<Id>, logic_op: LogicOp) -> Vec<Instruction> {
        let mut instructions = vec!();
        // if there are fewer src-operands than activated rows perform input replication
        row_addresses.sort_by_key(|row| ((ARCHITECTURE.get_distance_of_row_to_sense_amps)(*row))); // replicate input that resides in row with lowest success-rate (=probably the row furthest away)
        let nr_elements_to_extend = row_addresses.len() - src_operands.len();
        if nr_elements_to_extend > 0 {
            let last_element = *src_operands.last().unwrap();
            src_operands.extend( std::iter::repeat_n(last_element, nr_elements_to_extend));
        }

        for (&row_addr, &src_operand) in row_addresses.iter().zip(src_operands.iter()) {
            let src_operand_location = self.comp_state.value_states.get(&src_operand).expect("Src operand not available although it is used by a candidate. Sth went wrong...")
                .row_location.expect("Src operand not live although it is used by a candidate. Sth went wrong...");

            self.comp_state.dram_state.insert(row_addr, RowState { is_compute_row: true, live_value: Some(src_operand), constant: None });

            if (src_operand_location & SUBARRAY_ID_BITMASK) == (row_addr & SUBARRAY_ID_BITMASK) {
                instructions.push(Instruction::RowCloneFPM(src_operand_location, row_addr));
            } else {
                instructions.push(Instruction::RowClonePSM(src_operand_location, row_addr));
            }
        }
        instructions
    }
    /// Return id of subarray to use for computation and reference (compute_subarrayid, reference_subarrayid)
    /// - based on location of input rows AND current compilation state
    /// - [ ] POSSIBLE EXTENSION: include lookahead for future ops and their inputs they depend on
    fn select_compute_and_ref_subarray(&self, input_rows: Vec<RowAddress>) -> (SubarrayId, SubarrayId) {
        debug!("[ENTER select_compute_and_ref_subarray]");
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
        let mut next_instructions = vec!();

        debug!("Executing candidate {:?}", next_candidate);
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

        let logic_op = match language_op {
            Aig::And(_) => LogicOp::AND,
            Aig::Or(_) => LogicOp::OR,
            // TODO: extract NOT
            _ => panic!("candidate is expected to be a logic op"),
        };

        // 0. Select an SRA (=row-address tuple) for the selected subarray based on highest success-rate
        // TODO (possible improvement): input replication by choosing SRA with more activated rows than operands and duplicating operands which are in far-away rows into several rows?)
        let row_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&(nr_rows as u8)).unwrap()
            // sort by success-rate - using eg `BTreeMap` turned out to impose a too large runtime overhead
            .iter()
            .find(|combi| !self.blocked_row_combinations.contains(combi)) // choose first block RowAddr-combination
            .expect("No SRA for nr-rows={nr_rows}");

        let activated_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(row_combi).unwrap();
        let ref_rows: Vec<RowAddress> = activated_rows.iter()
            .map(|row| row & (subarrayid_to_subarray_address(ref_subarray))) // make activated rows refer to the right subarray
            .collect();
        let comp_rows: Vec<RowAddress>  = activated_rows.iter()
            .map(|row| row & (subarrayid_to_subarray_address(compute_subarray))) // make activated rows refer to the right subarray
            .collect();


        // 1. Initialize rows in ref-subarray (if executing AND/OR)
        // - TODO: read nr of frac-ops to issue from compiler-settings
        let mut instruction_init_ref_subarray  = self.init_reference_subarray(ref_rows.clone(), logic_op);
        next_instructions.append(&mut instruction_init_ref_subarray);

        // 2. Place rows in the simultaneously activated rows in the compute subarray (init other rows with 0 for OR, 1 for AND and same value for NOT)
        let mut instructions_init_comp_subarray = self.init_compute_subarray( activated_rows.clone(), src_operands, logic_op);
        next_instructions.append(&mut instructions_init_comp_subarray);

        // SKIPPED: 2.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
        // - only necessary once we find optimization to not write values to safe-space but reuse them diectly
        // 2.2.1 if yes: move data to other rows for performing this op

        // 3. Issue actual operation
        next_instructions.push(Instruction::APA(row_combi.0, row_combi.1));
        for row in comp_rows {
            self.comp_state.dram_state.insert(row, RowState { is_compute_row: true, live_value: Some(*next_candidate), constant: None });
        }
        // TODO NEXT: update `comp_state` (computed values, location of values, ..)

        // 4. Copy result data from dst NEAREST to the sense-amps into a safe-space row
        // TODO: possible improvement - error correction over all dst-rows (eg majority-vote for each bit, votes weighted by distance to sense-amps?)

        next_instructions
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
                            .unwrap_or(&ValueState{is_computed: false, row_location: None }) // no entry means this is the first time accessing this value
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
    /// Some rows (mostly only safe-space rows) store constants values, see [`CompilationState::constant_values`]
    constant: Option<usize>,
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
    /// Some constants are stored in fixed rows (!in each subarray), eg 0s and 1s for initializing reference subarray
    constant_values: HashMap<usize, RowAddress>,
    /// List of candidates (ops ready to be issued) prioritized by some metric by which they are scheduled for execution
    /// - NOTE: calculate Nodes `SchedulingPrio` using
    candidates: PriorityQueue<Id, SchedulingPrio>,
}

impl CompilationState {
    pub fn new(dram_state: HashMap<RowAddress, RowState>) -> Self {
        Self {
            dram_state,
            value_states: HashMap::new(),
            constant_values: HashMap::new(),
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
        Compiler::new(CompilerSettings { print_program: true, verbose: true, print_compilation_stats: false, min_success_rate: 0.999, repetition_fracops: 5, safe_space_rows_per_subarray: 16 } )
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
