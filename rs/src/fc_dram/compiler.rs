//!
//! - [`Compiler`] = performs actual compilation
//!     - [`CompilationState`] = stores states encountered during compilation (eg which values reside in which rows, rows containing live values, ..)
//!         - also see [`RowState`]
//! - [`SchedulingPrio`] = used to prioritize/order instruction for Instruction Scheduling
//!
//! - [`Compiler::compile()`] = main function - compiles given logic network for the given [`architecture`] into a [`program`] using some [`optimization`]

use super::{
    architecture::{subarrayid_to_subarray_address, Instruction, LogicOp, NeighboringSubarrayRelPosition, SubarrayId, SupportedNrOperands, ARCHITECTURE, NR_SUBARRAYS, ROW_ID_BITMASK, SUBARRAY_ID_BITMASK}, optimization::optimize, CompilerSettings, Program, RowAddress
};
use eggmock::{Aoig, Id, NetworkWithBackwardEdges, Node, Signal};
use itertools::Itertools;
use log::debug;
use priority_queue::PriorityQueue;
use strum::IntoEnumIterator;
use toml::{Table, Value};
use std::{cmp::Ordering, collections::{HashMap, HashSet}, env::consts::ARCH, ffi::CStr, fmt::Debug, fs, path::Path};

/// Provides [`Compiler::compile()`] to compile a logic network into a [`Program`]
pub struct Compiler {
    /// compiler-options set by user
    settings: CompilerSettings,
    /// Stores the state of all rows at each compilation step
    comp_state: CompilationState,
    /// These rows are reserved in EVERY subarray for storing intermediate results (ignore higher bits of these RowAddress)
    /// - NOTE: only initialized when compilation starts
    /// - NOTE2: all safe-space rows are treated equally, since no computation is performed on them (and hence distance to sense-amps doesn't really matter)
    ///
    /// DEPRECATED: use `blocked_row_combinations` instead (all other rows are safe-space rows)
    safe_space_rows: Vec<RowAddress>,
    /// RowCombinations which are not allowed to be issued via `APA` since they activate rows within the safe-space
    blocked_row_combinations: HashSet<(RowAddress,RowAddress)>,
    /// For each nr of operands this field store the rowaddress-combination to issue to activate
    /// the desired nr of rows (the choice is made best on success-rate and maximizing the nr of rows which potentially can't be used for storage
    /// since they would activate rows where values could reside in)
    compute_row_activations: HashMap<(SupportedNrOperands, NeighboringSubarrayRelPosition), (RowAddress,RowAddress)>
}

impl Compiler {
    pub fn new(settings: CompilerSettings) -> Self {
        Compiler{
           settings,
           comp_state: CompilationState::new( HashMap::new() ),
           safe_space_rows: vec!(),
           blocked_row_combinations: HashSet::new(),
           compute_row_activations: HashMap::new(),
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
        network: &impl NetworkWithBackwardEdges<Node = Aoig>,
    ) -> Program {
        let mut program = Program::new(vec!());

        // 0. Prepare compilation: select safe-space rows, place inputs into DRAM module (and store where inputs have been placed in `program`)
        self.init_comp_state(network, &mut program);

        // start with inputs
        let primary_inputs = network.leaves();
        debug!("Primary inputs: {:?}", primary_inputs.collect::<Vec<Id>>());

        // println!("{:?}", network.outputs().collect::<Vec<Signal>>());
        debug!("Nodes in network:");
        for node in network.iter() {
            debug!("{:?},", node);
        }

        // 1. Actual compilation
        while let Some((next_candidate, _)) = self.comp_state.candidates.pop() {
                // TODO: extend program with instr that is executed next
                let executed_instructions = &mut self.execute_next_instruction(&next_candidate, network);
                program.instructions.append(executed_instructions);

                // update new candidates (`next_candidate` is now available)
                let new_candidates = self.get_new_candidates(network, next_candidate);
                debug!("New candidates: {:?}", new_candidates);

                self.comp_state.candidates.extend(new_candidates);
        }

        // optimize(&mut program);

        // store output operand location so user can retrieve them after running the program
        let outputs = network.outputs();
        // TODO: doesn't work yet
        program.output_row_operands_placement = outputs.map(|out| {
            (out, self.comp_state.value_states.get(&out).unwrap().row_location.expect("ERROR: one of the outputs hasn't been computed yet..."))
        }).collect();
        program
    }

    /// Rather than making sure rows in which live values reside remain untouched, this approach chooses to select fixed RowAddress combinations for all support numbers of operands
    /// - this function sets [`Compiler::compute_row_activations`] to use as compute rows
    /// - NOTE: this choice is expected to be applicable to row activations in all subarrays since the SRA work equivalently between subarrays
    ///     - ASSUMPTION: there are no architectural differences btw subarrays
    ///
    /// # Limitations
    ///
    /// There are several drawbacks of choosing fixed compute rows:
    /// 1. *LogicOp* is not taken into consideration: different compute rows might (in theory) perform better for specific LogicOps (see [`LogicOp::get_success_rate_by_row_distance()`] which returns different SuccessRates based on the corresponding LogicOp)
    /// 2. Compute Rows might perform better for the next subarray (+1) than for the previous (-1) subarray (choice of subarray determines which SenseAmps are used and hence the distance btw rows and SenseAmps)
    ///
    /// This choice aims to finding a good compromise btw those limitations.
    /// TODO: NEXT
    fn choose_compute_rows(&mut self) {
        for nr_operands in SupportedNrOperands::iter() {
            for sense_amp_position in NeighboringSubarrayRelPosition::iter() {

                let possible_row_combis = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&nr_operands).expect("Given Architecture doesn't support SRA of {nr_operands} operands");
                let best_row_combi = possible_row_combis.iter().fold(possible_row_combis[0], |best_row_combi, next_row_combi| {
                    // compare row-combis based on avg success-rate and return the better one of them

                    let avg_distance_next_row_combi: u64 = {
                        let activated_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(next_row_combi).unwrap();
                        activated_rows.iter().map(|&row| {
                            // move subarray to 1st subarray (instead of 0th, which is at the edge and hence has no sense-amps above)
                            let subarray1_id = ((ROW_ID_BITMASK << 1) | 1) ^ ROW_ID_BITMASK;
                            let row =  subarray1_id | row; // makes sure that `get_distance_of_row_to_sense_amps` doesn't panic since SRA returns subarray=0 by default (which is an edge subarray)
                            println!("{row:b}");
                            (ARCHITECTURE.get_distance_of_row_to_sense_amps)(row, sense_amp_position) as u64
                        }).sum()
                    };
                    let avg_distance_best_row_combi: u64 = {
                        // move subarray to 1st subarray (instead of 0th, which is at the edge and hence has no sense-amps above)
                        let activated_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(&best_row_combi).unwrap();
                        activated_rows.iter().map(|&row| {
                            let subarray1_id = ((ROW_ID_BITMASK << 1) | 1) ^ ROW_ID_BITMASK;
                            let row =  subarray1_id | row; // makes sure that `get_distance_of_row_to_sense_amps` doesn't panic since SRA returns subarray=0 by default (which is an edge subarray)
                            println!("{row:b}");
                            (ARCHITECTURE.get_distance_of_row_to_sense_amps)(row, sense_amp_position) as u64
                        }).sum()
                    };

                    if avg_distance_next_row_combi > avg_distance_best_row_combi {
                        *next_row_combi
                    } else {
                        best_row_combi
                    }
                });

                self.compute_row_activations.insert((nr_operands, sense_amp_position), best_row_combi);
            }
        }
    }
    /// Allocates safe-space rows inside the DRAM-module
    /// - NOTE: nr of safe-space rows must be a power of 2 (x) between 1<=x<=64
    /// - [ ] TODO: improve algo (in terms of space efficiency)
    ///
    /// PROBLEM: can't test all possibilities since (for nr safe-space rows=16) there are  `math.comb(512,16) = 841141456821158064519401490400 = 8,4*10^{29}` of them
    fn alloc_safe_space_rows(&mut self, nr_safe_space_rows: u8) {

        let supported_nr_safe_space_rows = vec!(1,2,4,8,16,32,64);
        if !supported_nr_safe_space_rows.contains(&nr_safe_space_rows) {
            panic!("Only the following nr of rows are supported to be activated: {:?}, given: {}", supported_nr_safe_space_rows, nr_safe_space_rows);
        }

        // TODO: this is just a quick&dirty implementation. Solving this problem of finding optimal safe-space rows is probably worth solving for every DRAM-module once
        self.safe_space_rows = {

            // choose any row-addr combi activating exactly `nr_safe_space_rows` and choose all that activated rows to be safe-space rows
            let chosen_row_addr_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations
                .get(&SupportedNrOperands::try_from(nr_safe_space_rows).unwrap()).unwrap()
                .first().unwrap(); // just take the first row-combi that activates `nr_safe_space_rows`
            ARCHITECTURE.precomputed_simultaneous_row_activations.get(chosen_row_addr_combi).unwrap()
                .iter().map(|row| row & ROW_ID_BITMASK) // reset subarray-id to all 0s
                .collect()
        };

        // TODO: if any other rows are unusable as a consequence (bc activating them would activate another safe-space-row anytime) include those unusable rows into safe-space (else they're competely useless)
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
            // does it make sense to store any other constants in safe-space??

            self.comp_state.dram_state.insert(row_address_0, RowState { is_compute_row: false, live_value: None, constant: Some(0)} );
            self.comp_state.dram_state.insert(row_address_1, RowState { is_compute_row: false, live_value: None, constant: Some(1)} );
        }
    }

    /// Place inputs onto appropriate rows
    /// - NOTE: constants are expected to be placed before the inputs
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of move-ops to move intermediate results around & keep inputs close to sense-amps
    /// TODO: parititon logic network into subgraphs s.t. subgraphs can be mapped onto subarrays reducing nr of needed moves
    fn place_inputs(&mut self, mut inputs: Vec<Id>) {
        // naive implementation: start placing input on consecutive safe-space rows (continuing with next subarray once the current subarray has no more free safe-space rows)
        // TODO: change input operand placement to be more optimal (->taking into account future use of those operands)
        while let Some(next_input) = inputs.pop() {
            // NOTE: some safe-space rows are reserved for constants
            let free_safespace_row = self.get_next_free_safespace_row(None);

            let initial_row_state = RowState { is_compute_row: false, live_value: Some(Signal::new(next_input, false)), constant: None };
            let initial_value_state = ValueState { is_computed: true, row_location: Some(free_safespace_row) };
            self.comp_state.dram_state.insert(free_safespace_row, initial_row_state);
            self.comp_state.value_states.insert(Signal::new(next_input, false), initial_value_state);
        }
    }

    /// Initialize candidates with all nodes that are computable
    fn init_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>) {
        let inputs: Vec<Id> = network.leaves().collect();

        // init candidates with all nodes having only inputs as src-operands
        for &input in inputs.as_slice() {
            // every output has a prio determined eg by how many src-operands it uses last (->to minimize nr of live values in rows)
            let mut outputs_with_prio: PriorityQueue<Signal, SchedulingPrio> = network.node_outputs(input)
                .filter(|output| network.node(*output).inputs().iter().all(|other_input| inputs.contains(&other_input.node_id()) )) // only those nodes are candidates, whose src-operands are ALL inputs (->only primary inputs are directly available)
                .map( |output| {
                    let output_signal = Signal::new(output, false);
                    (output_signal, self.compute_scheduling_prio_for_node(output_signal, network))
                })
                .collect();
            self.comp_state.candidates.append(&mut outputs_with_prio);
            debug!("{:?} has the following outputs: {:?}", input, network.node_outputs(input).collect::<Vec<Id>>());
        }
    }

    /// Returns list of candidates that can be computed once `computed_node` is available
    fn get_new_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>, computed_node: Signal) -> PriorityQueue<Signal, SchedulingPrio> {
        debug!("Candidates: {:?}", self.comp_state.candidates);
        debug!("DRAM state: {:?}", self.comp_state.value_states);
        network.node_outputs(computed_node.node_id())
            // filter for new nodes that have all their input-operands available now (->only inputs of computed nodes could have changed to candidate-state, other nodes remain uneffected)
            .filter({|out| network.node(*out).inputs().iter()
                .all( |input| {
                    debug!("Out: {:?}, In: {:?}", out, input);
                    self.comp_state.value_states.keys().contains(input) &&  self.comp_state.value_states.get(input).unwrap().is_computed
                })
            })
            .map(|id| (Signal::new(id, false), self.compute_scheduling_prio_for_node(Signal::new(id, false), network))) // TODO: check if inverted signal is required as well!
            .collect()
    }

    /// Initialize compilation state: mark unsuable rows (eg safe-space rows), place input operands
    fn init_comp_state(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>, program: &mut Program) {
        let config_file = unsafe { CStr::from_ptr(self.settings.config_file) }.to_str().unwrap();
        let config = Path::new(config_file);
        println!("{:?}", config);

        // 0.1 Allocate safe-space rows (for storing intermediate values and constants 0s&1s) safely
        if config.is_file() {

            let content = fs::read_to_string(config).unwrap();
            let value = content.parse::<toml::Table>().unwrap(); // Parse into generic TOML Value :contentReference[oaicite:1]{index=1}

            if let Some(arr) = value.get("safe_space_rows").and_then(|v| v.as_array()) {
                println!("Found array of length {}", arr.len());
                    self.safe_space_rows = arr.iter().map(|v| {
                        v.as_integer().expect("Expected integer") as u64
                    }).collect();
            } else {
                panic!("Config file doesn't contain value for safe-space-rows");
            }

        } else {

            // debug!("Compiling {:?}", network);
            self.alloc_safe_space_rows(self.settings.safe_space_rows_per_subarray);

            // self.place_constants();

            let safe_space_rows_toml = Value::Array(self.safe_space_rows.iter().map(
                |row| Value::Integer(*row as i64)
            ).collect());
            let config_in_toml = toml::toml! {
                safe_space_rows = safe_space_rows_toml
            };
            fs::write(config, config_in_toml.to_string()).expect("Sth went wrong here..");
        }
        // TODO: allow reading these in from config-file
        self.choose_compute_rows(); // choose which rows will serve as compute rows (those are stored in `self.compute_row_activations`
        println!("{:?}", self.compute_row_activations);
        self.place_constants(); // TODO: move into config-file too?

        // deactivate all combination which could activate safe-space rows
        for row in self.safe_space_rows.iter() {
            for row_combi in ARCHITECTURE.row_activated_by_rowaddress_tuple.get(row).unwrap() {
                self.blocked_row_combinations.insert(*row_combi);
            }
        }

        // 0.2 Place all inputs and mark them as being live
        // TODO: add additional pass over graph: think about which inputs are needed in which subarray
        self.place_inputs( network.leaves().collect::<Vec<Id>>() ); // place input-operands into rows
        debug!("Placed inputs {:?} in {:?}", network.leaves().collect::<Vec<Id>>(), self.comp_state.value_states);
        // 0.3 Setup: store all network-nodes yet to be compiled
        self.init_candidates(network);

        // store where inputs have been placed in program for user to know where to put them when calling into this program
        program.input_row_operands_placement = network.leaves().collect::<Vec<Id>>()
            .iter()
            .flat_map(|&id| {
                let mut locations = Vec::new();
                let original_value = Signal::new(id, false);
                let inverted_value = Signal::new(id, false); // inverted value might have also been placed on init
                if let Some(value) = self.comp_state.value_states.get(&original_value) {
                    locations.push((original_value, value.row_location.expect("Inputs are init directly into rows at the start")));
                }

                if let Some(value) = self.comp_state.value_states.get(&inverted_value) {
                    locations.push((inverted_value, value.row_location.expect("Inputs are init directly into rows at the start")));
                }

                debug_assert_ne!(locations, vec!(), "Input {id:?} has not been placed at all");
                locations
            }).collect();
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
                    instructions.push(Instruction::RowCloneFPM(*row_address_1, other_row, String::from("Init ref-subarray with 1s")));
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
                    instructions.push(Instruction::RowCloneFPM(*row_address_0, other_row, String::from("Init ref-subarray with 0s")));
                }
                instructions
            },
            LogicOp::NOT => vec!(),
            _ => panic!("{logic_op:?} not supported yet"),
        }
    }

    /// Places the referenced `src_operands` into the corresponding `row_addresses` which are expected to be simultaneously executed using [`Instruction::RowCloneFPM`]
    /// - NOTE: `rel_pos_of_ref_subarray` might affect placement of inputs in the future (eg to choose which input rows to choose for *input replication*)
    fn init_compute_subarray(&mut self, mut row_addresses: Vec<RowAddress>, mut src_operands: Vec<Signal>, logic_op: LogicOp, rel_pos_of_ref_subarray: NeighboringSubarrayRelPosition) -> Vec<Instruction> {
        let mut instructions = vec!();
        // if there are fewer src-operands than activated rows perform input replication
        row_addresses.sort_by_key(|row| ((ARCHITECTURE.get_distance_of_row_to_sense_amps)(*row, rel_pos_of_ref_subarray.clone()))); // replicate input that resides in row with lowest success-rate (=probably the row furthest away)
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
                instructions.push(Instruction::RowCloneFPM(src_operand_location, row_addr, String::from("Move operand to compute row")));
            } else {
                instructions.push(Instruction::RowClonePSM(src_operand_location, row_addr)); // TODO: remove this, since it's not usable in COTS DRAMs
            }
        }
        instructions
    }

    /// Return sequence of instructions to provide negated inputs (if there are any among `src_operands`)).
    ///
    /// NOTE: Some inputs may be needed in a negated form by the candidates. To start execution those
    /// input operands have to be available with their negated form.
    /// TODO: NEXT
    fn init_negated_src_operands(&mut self, src_operands: Vec<Signal>, network: &impl NetworkWithBackwardEdges<Node = Aoig>) -> Vec<Instruction> {
        let mut instructions = vec!();
        let mut negated_inputs: HashSet<Signal> = HashSet::new(); // inputs which are required in their negated form
        let negated_src_operands: Vec<Signal> = src_operands.iter()
            .filter(|sig| sig.is_inverted())
            .copied() // map ref to owned val
            .collect();
        negated_inputs.extend(negated_src_operands.iter());

        for neg_in in negated_inputs {
            if self.comp_state.value_states.contains_key(&neg_in) {
                // negated signal is already available
                continue;
            } else {
                // else make negated-signal available
                let unnegated_signal = neg_in.invert();
                let origin_unneg_row= self.comp_state.value_states.get(&unnegated_signal).expect("Original version of this value is not available??")
                    .row_location.expect("Original version of this value is not live??");
                let dst_row = self.get_next_free_safespace_row(None);

                // TODO: negate val and move value in selected safe-space row
                let selected_sra = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&SupportedNrOperands::try_from(1).unwrap()).unwrap().iter().find(|row| ! self.safe_space_rows.contains(&row.0)) // just select the first available compute-row
                    .expect("It's assumed that issuing APA(row,row) for same row activates only that row");
                let origin_subarray_bitmask = origin_unneg_row & SUBARRAY_ID_BITMASK;
                let comp_row = origin_subarray_bitmask | (ROW_ID_BITMASK & selected_sra.0);
                let (_, ref_array) = self.select_compute_and_ref_subarray(vec!(comp_row));
                let result_row = (ROW_ID_BITMASK & selected_sra.0) & ref_array;

                let move_to_comp_row = Instruction::RowCloneFPM(origin_unneg_row, comp_row, String::from("Move row to safe space"));
                let not = Instruction::ApaNOT(comp_row,  result_row);
                let move_to_safespace= Instruction::RowCloneFPM(result_row, dst_row, String::from("Move row to safe space"));

                instructions.push(move_to_comp_row);
                instructions.push(not);
                instructions.push(move_to_safespace);

                self.comp_state.dram_state.insert(dst_row, RowState { is_compute_row: false, live_value: Some(neg_in), constant: None });
                self.comp_state.value_states.insert(neg_in, ValueState { is_computed: true, row_location: Some(dst_row) });
            }
        }

        instructions
    }

    /// Return id of subarray to use for computation and reference (compute_subarrayid, reference_subarrayid)
    /// - based on location of input rows AND current compilation state
    /// - [ ] POSSIBLE EXTENSION: include lookahead for future ops and their inputs they depend on
    ///
    /// TODO: NEXT
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

    /// Return next free safe-space row
    /// - use `preferred_subarray` if there is a specific subarray you would like to be that
    /// safe-sapce row from. Else just the next free safe-space row will be chosen
    ///     - NOTE: this is not guaranteed to be fulfilled !
    fn get_next_free_safespace_row(&self, preferred_subarray: Option<SubarrayId>) -> RowAddress {

        let subarray_order  =  if let Some(subarray) = preferred_subarray {
            // start search with `preferred_subarray` if it's supplied
            let mut first_half = (0..subarray).collect::<Vec<SubarrayId>>();
            first_half.extend(subarray+1..NR_SUBARRAYS);
            first_half
            // subarray_original_order.filter(|x| *x != subarray).into_iter()
        } else {
            // else just iterate in natural order
            (0..NR_SUBARRAYS).collect()
        };

        for subarray in subarray_order {
            for row in &self.safe_space_rows {
                let row_addr = row | subarrayid_to_subarray_address(subarray);
                if self.comp_state.dram_state.get(&row_addr).unwrap_or(&RowState::default()).live_value.is_none() { // NOTE: safe-space rows are inserted lazily into `dram_state`
                    return row_addr;
                }
            }
        }
        panic!("OOM: No more available safe-space rows");
    }

    /// Returns Instructions to execute given `next_candidate`
    /// - [ ] make sure that operation to be executed on those rows won't simultaneously activate other rows holding valid data which will be used by future operations
    /// TODO: NEXT
    fn execute_next_instruction(&mut self, next_candidate: &Signal, network: &impl NetworkWithBackwardEdges<Node = Aoig>) -> Vec<Instruction> {
        let mut next_instructions = vec!();

        debug!("Executing candidate {:?}", next_candidate);
        let src_operands: Vec<Signal> = network.node(next_candidate.node_id()).inputs().to_vec();
        let mut init_neg_operands = self.init_negated_src_operands(src_operands.clone(), network); // TODO NEXT: make sure all required negated operands are available
        next_instructions.append(&mut init_neg_operands);

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
        let language_op = network.node(next_candidate.node_id());

        // TODO: extract NOT
        let logic_op = match language_op {
            // REMINDER: operand-nr is extracted by looking at nr of children beforehand
            Aoig::And(_) | Aoig::And4(_) | Aoig::And8(_)| Aoig::And16(_) => LogicOp::AND,
            Aoig::Or(_) | Aoig::Or4(_) | Aoig::Or8(_) | Aoig::Or16(_) => LogicOp::OR,
            _ => panic!("candidate is expected to be a logic op"),
        };

        // 0. Select an SRA (=row-address tuple) for the selected subarray based on highest success-rate
        // TODO (possible improvement): input replication by choosing SRA with more activated rows than operands and duplicating operands which are in far-away rows into several rows?)
        let row_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&SupportedNrOperands::try_from(nr_rows as u8).unwrap()).unwrap()
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
        let mut instructions_init_comp_subarray = self.init_compute_subarray( activated_rows.clone(), src_operands, logic_op, NeighboringSubarrayRelPosition::get_relative_position(compute_subarray, ref_subarray));
        next_instructions.append(&mut instructions_init_comp_subarray);

        // SKIPPED: 2.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
        // - only necessary once we find optimization to not write values to safe-space but reuse them diectly
        // 2.2.1 if yes: move data to other rows for performing this op

        // 3. Issue actual operation
        let mut actual_op = match logic_op {
            LogicOp::NOT => vec!(Instruction::ApaNOT(row_combi.0, row_combi.1)),
            LogicOp::AND | LogicOp::OR => vec!(Instruction::ApaAndOr(row_combi.0, row_combi.1)),
            LogicOp::NAND | LogicOp::NOR => vec!(Instruction::ApaAndOr(row_combi.0, row_combi.1), Instruction::ApaNOT(row_combi.0, row_combi.1)), // TODO: or the othyer way around (1st NOT)?
        };

        next_instructions.append(&mut actual_op);
        for (&comp_row, &ref_row) in comp_rows.iter().zip(ref_rows.iter()) {
            self.comp_state.dram_state.insert(comp_row, RowState { is_compute_row: true, live_value: Some(*next_candidate), constant: None });
            self.comp_state.value_states.insert(*next_candidate, ValueState { is_computed: true, row_location: Some(comp_row) });
            // ref subarray holds negated value afterwarsd
            self.comp_state.dram_state.insert(ref_row, RowState { is_compute_row: true, live_value: Some(next_candidate.invert()), constant: None });
            self.comp_state.value_states.insert(next_candidate.invert(), ValueState { is_computed: true, row_location: Some(ref_row) });
        }

        // 4. Copy result data from dst NEAREST to the sense-amps into a safe-space row and update `value_state`
        // TODO LAST: possible improvement - error correction over all dst-rows (eg majority-vote for each bit, votes weighted by distance to sense-amps?)

        next_instructions
    }

    /// Compute `SchedulingPrio` for a given node
    /// - used for inserting new candidates
    /// TODO: write unittest for this function
    fn compute_scheduling_prio_for_node(&self, signal: Signal, network: &impl NetworkWithBackwardEdges<Node = Aoig>) -> SchedulingPrio {
        let nr_last_value_uses = network.node(signal.node_id()).inputs() // for each input check whether `id` is the last node using it
            .iter()
            .fold(0, |acc, input| {
                let input_id = Signal::node_id(input);
                let non_computed_outputs: Vec<Id> = network.node_outputs(input_id) // get all other nodes relying on this input
                    .filter(|out| {

                        let out_signal = Signal::new(*out,false);
                        out_signal != signal &&
                        !(self.comp_state.value_states.get(&out_signal)
                            .unwrap_or(&ValueState{is_computed: false, row_location: None }) // no entry means this is the first time accessing this value
                            .is_computed)
                    }
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
            nr_src_operands: network.node(signal.node_id()).inputs().len(),
            nr_result_operands:  network.node_outputs(signal.node_id()).collect::<Vec<Id>>().len(),
        }
    }
}

/// Stores the current state of a row at a concrete compilations step
#[derive(Default)] // by default not a compute_row, no live-value and no constant inside row
pub struct RowState {
    /// True iff that row is currently: 1) Not a safe-sapce row, 2) Doesn't activate any safe-sapce rows, 3) Isn't holding valid values in the role of a reference-subarray row
    is_compute_row: bool,
    /// `None` if the value inside this row is currently not live
    live_value: Option<Signal>,
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
    value_states: HashMap<Signal, ValueState>,
    /// Some constants are stored in fixed rows (!in each subarray), eg 0s and 1s for initializing reference subarray
    constant_values: HashMap<usize, RowAddress>,
    /// List of candidates (ops ready to be issued) prioritized by some metric by which they are scheduled for execution
    /// - NOTE: calculate Nodes `SchedulingPrio` using
    candidates: PriorityQueue<Signal, SchedulingPrio>,
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
    use eggmock::{AoigLanguage, ComputedNetworkWithBackwardEdges, Network};

    use crate::fc_dram::cost_estimation::CompilingCostFunction;

    use super::*; use std::ffi::CString;
    // import all elements from parent-module
    use std::sync::Once;

    // ERROR: `eggmock`-API doesn't allow this..
    // // For data shared among unittests but initalized only once
    // static TEST_DATA: LazyLock<_> = LazyLock::new(|| {
    //     let mut egraph: EGraph<AoigLanguage, ()> = Default::default();
    //     let my_expression: RecExpr<AoigLanguage> = "(and (and a c) (and b c))".parse().unwrap();
    //     let extractor = Extractor::new( &egraph, CompilingCostFunction {});
    //     let ntk = (extractor, vec!(egg::Id::from(9)));
    //
    //     ComputedNetworkWithBackwardEdges::new(&ntk)
    // });

    // ERROR: This also does not work bc of the weird implementation of a network
    // fn simple_egraph() -> ComputedNetworkWithBackwardEdges<'static, (Extractor<'static, CompilingCostFunction, AoigLanguage, ()>, Vec<egg::Id>)> {
    //     let mut egraph: EGraph<AoigLanguage, ()> = Default::default();
    //     let my_expression: RecExpr<AoigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
    //     egraph.add_expr(&my_expression);
    //     let extractor = Extractor::new( &egraph, CompilingCostFunction {});
    //     let ntk = &(extractor, vec!(egg::Id::from(5)));
    //     ntk.dump();
    //     // Id(5): And([Signal(false, Id(2)), Signal(false, Id(4))])
    //     // Id(4): And([Signal(false, Id(3)), Signal(false, Id(1))])
    //     // Id(1): Input(3)
    //     // Id(3): Input(2)
    //     // Id(2): And([Signal(false, Id(0)), Signal(false, Id(1))])
    //     // Id(0): Input(1)
    //
    //     let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);
    //     ntk_backward
    // }

    static INIT: Once = Once::new();

    fn init() -> Compiler {
        INIT.call_once(|| {
            env_logger::init();
        });
        Compiler::new(CompilerSettings { print_program: true, verbose: true, print_compilation_stats: false, min_success_rate: 0.999, repetition_fracops: 5, safe_space_rows_per_subarray: 16, config_file: CString::new("").expect("CString::new failed").as_ptr(), do_save_config: true} )
    }

    #[test]
    fn test_alloc_safe_space_rows() {
        let mut compiler = init();
        const REQUESTED_SAFE_SPACE_ROWS: u8 = 8;
        compiler.alloc_safe_space_rows(REQUESTED_SAFE_SPACE_ROWS);

        debug!("{:?}", compiler.safe_space_rows);
        assert_eq!(compiler.safe_space_rows.iter().dedup().collect::<Vec<&RowAddress>>().len(), REQUESTED_SAFE_SPACE_ROWS as usize);
    }

    #[test]
    fn test_candidate_initialization() {
        let mut compiler = init();

        let mut egraph: EGraph<AoigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AoigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
        egraph.add_expr(&my_expression);
        let output2 = egraph.add(AoigLanguage::And([eggmock::egg::Id::from(0), eggmock::egg::Id::from(2)])); // additional `And` with one src-operand=input and one non-input src operand
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
        compiler.comp_state.value_states.insert(Signal::new(eggmock::Id::from(4), false), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        compiler.init_candidates(&ntk_backward);
        let is_candidate_ids: HashSet<Signal> = compiler.comp_state.candidates.iter().map(|(id,_)| *id).collect();
        let should_candidate_ids: HashSet<Signal> = HashSet::from([Signal::new( eggmock::Id::from(2), false), Signal::new(eggmock::Id::from(4), false)]);
        assert_eq!( is_candidate_ids, should_candidate_ids);

        // TODO: test-case with node that relies on one input src-operand and one non-input (intermediate node) src-operand
    }

    #[test]
    fn test_new_candidates() {
        let mut compiler = init();

    }

    #[test]
    fn test_compute_scheduling_prio_for_node() {
        let mut compiler = init();

        let mut egraph: EGraph<AoigLanguage, ()> = Default::default();
        let my_expression: RecExpr<AoigLanguage> = "(and (and 1 3) (and 2 3))".parse().unwrap();
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
        compiler.comp_state.value_states.insert(Signal::new(eggmock::Id::from(2), false), ValueState{ is_computed: true, row_location: None });

        let ntk_backward = ComputedNetworkWithBackwardEdges::new(ntk);

        let scheduling_prio = compiler.compute_scheduling_prio_for_node(Signal::new(eggmock::Id::from(4), false),  &ntk_backward);
        assert_eq!(scheduling_prio, SchedulingPrio { nr_last_value_uses: 2, nr_src_operands: 2, nr_result_operands: 1 }  );

    }

    #[test]
    fn test_select_compute_and_ref_subarray() {
        let compiler = init();
        let (selected_subarray, _) = compiler.select_compute_and_ref_subarray(vec!(0b1_000_000_000, 0b1_000_010_000, 0b111_000_000_000, 0b10_100_000_000,));
        assert_eq!(selected_subarray, 0b1_000_000_000);
    }

    #[ignore]
    fn test_program_validity() {
        // 1. test that no APA activates a safe-space row

        // ..
    }
}
