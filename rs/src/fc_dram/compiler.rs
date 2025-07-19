//!
//! - [`Compiler`] = performs actual compilation
//!     - [`CompilationState`] = stores states encountered during compilation (eg which values reside in which rows, rows containing live values, ..)
//!         - also see [`RowState`]
//! - [`SchedulingPrio`] = used to prioritize/order instruction for Instruction Scheduling
//!
//! - [`Compiler::compile()`] = main function - compiles given logic network for the given [`architecture`] into a [`program`] using some [`optimization`]

use crate::fc_dram::architecture::ROWS_PER_SUBARRAY;

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
    /// For each nr of operands this field store the rowaddress-combination to issue to activate
    /// the desired nr of rows (the choice is made best on success-rate and maximizing the nr of rows which potentially can't be used for storage
    /// since they would activate rows where values could reside in)
    /// - This is a Design Decision taken: compute rows are rows reserved for performing computations, all other rows are usable as "Register"
    compute_row_activations: HashMap<(SupportedNrOperands, NeighboringSubarrayRelPosition), (RowAddress,RowAddress)>,
    /// Stores all subarrays in which the signal has to be available
    signal_to_subarrayids: HashMap<Signal, Vec<SubarrayId>>,
    /// see [`Self::get_all_noninverted_src_signals`]. First `Vec<Signal>`=noninverted src signals, 2nd `Vec<Signal>`=inverted src signals
    computed_noninverted_scr_signals: HashMap<Signal, (Vec<Signal>,Vec<Signal>)>,
}

impl Compiler {
    /// Constants are repeated to fill complete row
    const CONSTANTS: [usize; 2] = [0, 1];

    pub fn new(settings: CompilerSettings) -> Self {
        Compiler{
           settings,
           comp_state: CompilationState::new( HashMap::new() ),
           compute_row_activations: HashMap::new(),
           signal_to_subarrayids: HashMap::new(),
           computed_noninverted_scr_signals: HashMap::new(),
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

        // debug!("Compiling {:?}", network);
        // 0. Prepare compilation:
        // - select safe-space rows
        // - place inputs&constants into DRAM module (and store where inputs have been placed in `program`)
        // - initialize candidates (with which to start execution)
        self.init_comp_state(network, &mut program);

        // println!("{:?}", network.outputs().collect::<Vec<Signal>>());
        // debug!("Nodes in network:");
        // for node in network.iter() {
        //     debug!("{:?},", node);
        // }

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
                            let row =  RowAddress(subarray1_id | row.0); // makes sure that `get_distance_of_row_to_sense_amps` doesn't panic since SRA returns subarray=0 by default (which is an edge subarray)
                            println!("{:b}", row.0);
                            (ARCHITECTURE.get_distance_of_row_to_sense_amps)(row, sense_amp_position) as u64
                        }).sum()
                    };
                    let avg_distance_best_row_combi: u64 = {
                        // move subarray to 1st subarray (instead of 0th, which is at the edge and hence has no sense-amps above)
                        let activated_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(&best_row_combi).unwrap();
                        activated_rows.iter().map(|&row| {
                            let subarray1_id = ((ROW_ID_BITMASK << 1) | 1) ^ ROW_ID_BITMASK;
                            let row =  RowAddress(subarray1_id | row.0); // makes sure that `get_distance_of_row_to_sense_amps` doesn't panic since SRA returns subarray=0 by default (which is an edge subarray)
                            println!("{:b}", row.0);
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

    /// Places (commonly used) constants in safe-space rows
    /// - ! all safe-space rows are assumed to be empty when placing constants (constans are the first things to be placed into safe-space rows)
    /// - currently placed constants: all 0s and all 1s (for [`Compiler::init_reference_subarray`]
    /// - TODO: store placement of constants in `program`
    fn place_constants(&mut self, program: &mut Program) {
        // place constants in EVERY subarray
        for subarray in 0..NR_SUBARRAYS {
            for constant in Self::CONSTANTS {
                let next_free_row = self.comp_state.free_rows_per_subarray.get_mut(&subarray).and_then(|v| v.pop()).expect("No free rows in subarray {subarray} :(");
                self.comp_state.constant_values.insert(constant, next_free_row);
                self.comp_state.dram_state.insert(next_free_row, RowState { is_compute_row: false, live_value: None, constant: Some(constant)} );
            }
        }
    }

    /// Place inputs onto appropriate rows, storing the decided placement into `program.input_row_operands_placement`
    /// - NOTE: constants are expected to be placed before the inputs
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of move-ops to move intermediate results around & keep inputs close to sense-amps
    /// TODO: parititon logic network into subgraphs s.t. subgraphs can be mapped onto subarrays reducing nr of needed moves
    fn place_inputs(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>, program: &mut Program) {

        for input in network.leaves().collect::<Vec<Id>>() {
            // check whether the signal is required in inverted or noninverted form and place it accordingly
            let original_signal = Signal::new(input, false);
            let inverted_signal = Signal::new(input, true);

            if let Some(original_input_locations) = self.signal_to_subarrayids.get(&original_signal) {
                for subarray in original_input_locations {
                    let next_free_row = self.comp_state.free_rows_per_subarray.get_mut(subarray).and_then(|v| v.pop()).expect("OOM: No more free rows in subarray {subarray} for placing inputs");
                    self.comp_state.value_states.insert(original_signal, ValueState { is_computed: true, row_location: Some(next_free_row) });

                    program.input_row_operands_placement.entry(original_signal).or_default().push(next_free_row);
                }
            }

            if let Some(inverted_input_locations) = self.signal_to_subarrayids.get(&inverted_signal) {
                for subarray in inverted_input_locations {
                    let next_free_row = self.comp_state.free_rows_per_subarray.get_mut(subarray).and_then(|v| v.pop()).expect("OOM: No more free rows in subarray {subarray} for placing inputs");
                    self.comp_state.value_states.insert(inverted_signal, ValueState { is_computed: true, row_location: Some(next_free_row) });

                    program.input_row_operands_placement.entry(inverted_signal).or_default().push(next_free_row);
                }
            }
        }
    }

    /// Initialize candidates with all nodes that are computable
    fn init_candidates(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>) {
        todo!("NEXT");
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

    /// Initialize compilation state:
    /// - choose compute rows (by setting [`Self::compute_row_activations`]
    /// - assign subarray-ids to each NodeId
    /// - initialize [`Self::comp_state::free_rows_per_subarray`] with the rows that are free to be used for placing constants, inputs and intermediate values (when execution has started)
    /// - return code to place input operands in `program`
    fn init_comp_state(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>, program: &mut Program) {
        let config_file = unsafe { CStr::from_ptr(self.settings.config_file) }.to_str().unwrap();
        let config = Path::new(config_file);
        println!("{:?}", config);

        // 0.1 Allocate compute rows: rows reserved for performing computations, all other rows are usable as "Register"
        if config.is_file() {

            // let content = fs::read_to_string(config).unwrap();
            // let value = content.parse::<toml::Table>().unwrap(); // Parse into generic TOML Value :contentReference[oaicite:1]{index=1}
            //
            // if let Some(arr) = value.get("safe_space_rows").and_then(|v| v.as_array()) {
            //     println!("Found array of length {}", arr.len());
            //         self.safe_space_rows = arr.iter().map(|v| {
            //             v.as_integer().expect("Expected integer") as u64
            //         }).collect();
            // } else {
            //     panic!("Config file doesn't contain value for safe-space-rows");
            // }

            // TODO: read&write this to&from config-file (added manually here in the meantiem)
            self.compute_row_activations = HashMap::from([
                ((SupportedNrOperands::One, NeighboringSubarrayRelPosition::Above), (RowAddress(8), RowAddress(8))),
                ((SupportedNrOperands::One, NeighboringSubarrayRelPosition::Below), (RowAddress(303), RowAddress(303))),
                ((SupportedNrOperands::Two, NeighboringSubarrayRelPosition::Above), (RowAddress(15), RowAddress(79))),
                ((SupportedNrOperands::Two, NeighboringSubarrayRelPosition::Below), (RowAddress(293), RowAddress(357))),
                ((SupportedNrOperands::Four, NeighboringSubarrayRelPosition::Above), (RowAddress(60), RowAddress(42))),
                ((SupportedNrOperands::Four, NeighboringSubarrayRelPosition::Below), (RowAddress(472), RowAddress(412))),
                ((SupportedNrOperands::Eight, NeighboringSubarrayRelPosition::Above), (RowAddress(42), RowAddress(15))),
                ((SupportedNrOperands::Eight, NeighboringSubarrayRelPosition::Below), (RowAddress(203), RowAddress(283))),
                ((SupportedNrOperands::Sixteen, NeighboringSubarrayRelPosition::Above), (RowAddress(32), RowAddress(83))),
                ((SupportedNrOperands::Sixteen, NeighboringSubarrayRelPosition::Below), (RowAddress(470), RowAddress(252))),
                ((SupportedNrOperands::Thirtytwo, NeighboringSubarrayRelPosition::Above), (RowAddress(307), RowAddress(28))),
                ((SupportedNrOperands::Thirtytwo, NeighboringSubarrayRelPosition::Below), (RowAddress(149), RowAddress(318))),
            ]);


        } else {
            self.choose_compute_rows(); // choose which rows will serve as compute rows (those are stored in `self.compute_row_activations`
            println!("{:?}", self.compute_row_activations);

            // TODO: write chosen compute rows to config-file
            // let safe_space_rows_toml = Value::Array(self.safe_space_rows.iter().map(
            //     |row| Value::Integer(*row as i64)
            // ).collect());
            // let config_in_toml = toml::toml! {
            //     safe_space_rows = safe_space_rows_toml
            // };
            // fs::write(config, config_in_toml.to_string()).expect("Sth went wrong here..");
        }

        // 0.2 Save free rows
        // At the start all rows, except for the compute rows, are free rows
        let compute_rows = self.compute_row_activations.values().fold(vec!(), |all_compute_rows, next_compute_row_combi| {
            let new_compute_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(next_compute_row_combi).expect("Compute row cant be activated??");
            all_compute_rows.iter().chain(new_compute_rows).cloned().collect()
        });
        let mut free_rows = (0..ROWS_PER_SUBARRAY).map(RowAddress::from).collect::<Vec<RowAddress>>();
        free_rows.retain(|r| {!compute_rows.contains(r)});
        for subarray in 0..NR_SUBARRAYS {
            let free_rows_in_subarray = free_rows.iter().map(|row| RowAddress(row.0 | subarrayid_to_subarray_address(subarray).0)).collect(); // transform local row address to row addresses in corresponding `subarray`
            self.comp_state.free_rows_per_subarray.entry(subarray as SubarrayId).insert_entry(free_rows_in_subarray);
        }

        // 0.3 Group operands by subarray (ensure all operands are placed in the right subarray)
        self.assign_signals_to_subarrays(network); // sets `self.signal_to_subarrayids`

        // NEXT: 0.4 Place all constants and inputs and mark the inputs as being live
        self.place_constants(program); // constants are placed in <u>each</u> subarray
        debug!("Placed constants: {:?}", self.comp_state.constant_values);

        self.place_inputs(network, program); // place input-operands into rows
        debug!("Placed inputs {:?} in {:?}", network.leaves().collect::<Vec<Id>>(), self.comp_state.value_states);

        // 0.5 Setup: store all network-nodes yet to be compiled
        self.init_candidates(network);
        debug!("Initialized candidates {:?}", self.comp_state.candidates);
    }

    /// Assigns signals to subarrays and through this determines placement of those signal in the DRAM module
    /// - sets [`Self::signal_to_subarrayids`]
    ///
    /// # Assumptions
    ///
    /// - assumes `network` is acyclic !
    ///
    /// # TODO
    ///
    /// - make sure nr of signals placed in a subarray is <= nr of available rows (without compute rows)
    /// - think about merging subarray assignment (s.t. several outputs end up in same subarray, so that inputs can be reused among them)
    fn assign_signals_to_subarrays(&mut self, network: &impl NetworkWithBackwardEdges<Node = Aoig>) {
        // 1. determine all signals which go into outputs without being negated (and hence can be stored in same subarray)
        //  - also store signals which do need to be negated (and process them in the next step)
        //  - TODO: continue graph traversal with src-operands of the outputs (until primary inputs are reached)
        let mut subarray_id = 1; // start with 1 since edge subarrays cant be used as compute subarrays
        for output in network.outputs() {
            self.signal_to_subarrayids.insert(output, vec!(subarray_id)); // determine (virtual) subarray in which output will reside
            let (noninverted_src_signals, inverted_src_signals) = self.get_all_noninverted_src_signals(output, network);

            println!("{:?}", noninverted_src_signals.clone());
            // all directly (might in theory) reside in the same subarray as `output` (since no NOTS are inbtw which locate them to a neighboring subarray)
            for connected_signal in noninverted_src_signals {
                self.signal_to_subarrayids.entry(connected_signal).or_default().push(subarray_id); // determine (virtual) subarray in which output will reside
            }

            // place inverted signals in neighboring subarray
            let neighboring_subarray = subarray_id - 1; // place signals that are inverted odd number of times in Above subarray (arbitrary decision, epxloring whether this makes a difference might be explored in future)
            let mut unvisited_signals_in_same_subarray: Vec<Signal> = vec!(); // inverting even nr of times leads to signals being placed in same subarray
            let mut unvisited_signals_in_neighboring_subarray = inverted_src_signals;
            while !unvisited_signals_in_same_subarray.is_empty() || !unvisited_signals_in_neighboring_subarray.is_empty() {
                // println!("Same subarray: {:?}", unvisited_signals_in_same_subarray);
                // println!("Neighboring subarray: {:?}", unvisited_signals_in_neighboring_subarray);
                if let Some(signal_neighboring_subarray) = unvisited_signals_in_neighboring_subarray.pop() {

                    self.signal_to_subarrayids.entry(signal_neighboring_subarray).or_default().push(neighboring_subarray);
                    // these are placed in the Above subarray (arbitrary decision, epxloring whether this makes a difference might be explored in future)
                    let (signals_neighboring_subarray_of_output,  mut signals_same_subarray_as_output) = self.get_all_noninverted_src_signals(signal_neighboring_subarray, network);
                    for inverted_signal in signals_neighboring_subarray_of_output {
                        self.signal_to_subarrayids.entry(inverted_signal).or_default().push(neighboring_subarray);
                    }

                    unvisited_signals_in_same_subarray.append(&mut signals_same_subarray_as_output);
                }

                if let Some(signal_same_subarray) = unvisited_signals_in_same_subarray.pop() {

                    self.signal_to_subarrayids.entry(signal_same_subarray).or_default().push(subarray_id);
                    // signals inverted even nr of times are placed in the same subarray as the `output` Signal
                    let (signals_same_subarray_of_output,  mut signals_neighboring_subarray_as_output) = self.get_all_noninverted_src_signals(signal_same_subarray, network);
                    for signal in signals_same_subarray_of_output {
                        self.signal_to_subarrayids.entry(signal).or_default().push(subarray_id);
                    }

                    unvisited_signals_in_neighboring_subarray.append(&mut signals_neighboring_subarray_as_output);
                }
            }

            // for the beginning place all outputs in different subarrays. A 2nd pass may optimize/merge subarrays later on
            subarray_id += 2; // maybe +=2 to account for negated operands being stored in neighboring subarray? (TODO: test with some example networks)
        }


        debug!("{:?}", self.signal_to_subarrayids);
    }

    /// Returns all src signals which are not inverted. These are exactly those signals that can be placed in the same subarray as.
    ///
    /// # Returns
    ///
    /// Tuple of
    /// 1. Vector of src Signals that are **not** inverted
    /// 2. Vector of src Signals that are indeed inverted (need to be processed further, only first inverted signal is returned for a subtree)
    fn get_all_noninverted_src_signals(&mut self, signal: Signal, network: &impl NetworkWithBackwardEdges<Node = Aoig>) -> (Vec<Signal>, Vec<Signal>) {
        let signal_node = network.node(signal.node_id());

        let mut noninverted_src_signals = vec!();
        let mut inverted_src_signals = vec!();
        let mut stack_unvisited_noninverted_src_operands = Vec::from(signal_node.inputs());

        while let Some(src_operand) = stack_unvisited_noninverted_src_operands.pop() {
            if src_operand.is_inverted() {
                inverted_src_signals.push(src_operand); // store subarray to which this input has to be placed to as a neighbor, further processing elsewhere
            } else {
                noninverted_src_signals.push(src_operand);
                let src_operand_node = network.node(src_operand.node_id());
                stack_unvisited_noninverted_src_operands.append(&mut Vec::from(src_operand_node.inputs()));
            }
        }

        self.computed_noninverted_scr_signals.insert(signal, (inverted_src_signals.clone(), noninverted_src_signals.clone())); // to save (possible) recomputation next time
        (noninverted_src_signals, inverted_src_signals)
    }

    /// Returns instructions to initialize all given `ref_rows` in reference-subarray for corresponding logic-op
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

            if (src_operand_location.0 & SUBARRAY_ID_BITMASK) == (row_addr.0 & SUBARRAY_ID_BITMASK) {
                instructions.push(Instruction::RowCloneFPM(src_operand_location, row_addr, String::from("Move operand to compute row")));
            } else {
                instructions.push(Instruction::RowClonePSM(src_operand_location, row_addr)); // TODO: remove this, since it's not usable in COTS DRAMs
            }
        }
        instructions
    }

    /// Returns instructions to be executed for performing `NOT` on `src_row` into `dst_row`
    /// - NOTE: currenlty only single-operand NOTs are supported bc
    ///     1) more operands lead to (slightly) worse results (see Figure10 in [1])
    ///     2) since there are separate compute rows using multiple dst rows doesn't make sense (the values need to be copied out of the dst-rows anyway into non-compute rows)
    fn execute_not(&self, signal_to_invert: &Signal) -> Vec<Instruction> {
        let row_combi = self.compute_row_activations.get(&(SupportedNrOperands::One, NeighboringSubarrayRelPosition::Above)).unwrap();
        // perform `NOT`

        let row_combi_correct_subarray = todo!(); // REMINDER: rows returned `compute_row_activations` are not yet adjusted for the right subarray
    }

    /// Returns the instructions needed to perform `language_op`
    fn execute_and_or(&self, language_op: Aoig) -> Vec<Instruction> {
        let logic_op = match language_op {
            // REMINDER: operand-nr is extracted by looking at nr of children beforehand
            Aoig::And(_) | Aoig::And4(_) | Aoig::And8(_)| Aoig::And16(_) => LogicOp::AND,
            Aoig::Or(_) | Aoig::Or4(_) | Aoig::Or8(_) | Aoig::Or16(_) => LogicOp::OR,
            _ => panic!("candidate is expected to be a logic op"),
        };

        todo!();
    }

    /// Returns Instructions to execute given `next_candidate`
    /// - [ ] make sure that operation to be executed on those rows won't simultaneously activate other rows holding valid data which will be used by future operations
    ///
    /// TODO: NEXT
    fn execute_next_instruction(&mut self, next_candidate: &Signal, network: &impl NetworkWithBackwardEdges<Node = Aoig>) -> Vec<Instruction> {
        let node_id = next_candidate.node_id();
        assert!(network.node(node_id).inputs().iter().all(|input| {
            self.signal_to_subarrayids.get(input).unwrap() == self.signal_to_subarrayids.get(&Signal::new(node_id, false))
        }));

        let mut next_instructions = vec!();
        // 1. Perform actual operation of the node
        let language_op = network.node(node_id);
        next_instructions.append(&mut self.execute_and_or(language_op));
        // 2. Negate the result (if needed)
        if next_candidate.is_inverted() {
            let mut negate_instructions = self.execute_not(next_candidate);
            next_instructions.append(&mut negate_instructions);
        }
        todo!();

        //
        // debug!("Executing candidate {:?}", next_candidate);
        // let src_operands: Vec<Signal> = network.node(next_candidate.node_id()).inputs().to_vec();
        // let mut init_neg_operands = self.init_negated_src_operands(src_operands.clone(), network); // TODO NEXT: make sure all required negated operands are available
        // next_instructions.append(&mut init_neg_operands);
        //
        // let nr_operands = src_operands.len(); // use to select SRA to activate
        // let nr_rows = nr_operands.next_power_of_two();
        //
        // let src_rows: Vec<RowAddress> = src_operands.iter()
        //     .map(|src_operand| {
        //
        //         debug!("src: {src_operand:?}");
        //         self.comp_state.value_states.get(src_operand)
        //             .unwrap()
        //             .row_location
        //             .expect("Sth went wrong... if the src-operand is not in a row, then this candidate shouldn't have been added to the list of candidates")
        //     })
        //     .collect();
        //
        // let (compute_subarray, ref_subarray) = self.select_compute_and_ref_subarray(src_rows);
        // let language_op = network.node(next_candidate.node_id());
        //
        // // TODO: extract NOT
        // let logic_op = match language_op {
        //     // REMINDER: operand-nr is extracted by looking at nr of children beforehand
        //     Aoig::And(_) | Aoig::And4(_) | Aoig::And8(_)| Aoig::And16(_) => LogicOp::AND,
        //     Aoig::Or(_) | Aoig::Or4(_) | Aoig::Or8(_) | Aoig::Or16(_) => LogicOp::OR,
        //     _ => panic!("candidate is expected to be a logic op"),
        // };
        //
        // // 0. Select an SRA (=row-address tuple) for the selected subarray based on highest success-rate
        // // TODO (possible improvement): input replication by choosing SRA with more activated rows than operands and duplicating operands which are in far-away rows into several rows?)
        // let row_combi = ARCHITECTURE.sra_degree_to_rowaddress_combinations.get(&SupportedNrOperands::try_from(nr_rows as u8).unwrap()).unwrap()
        //     // sort by success-rate - using eg `BTreeMap` turned out to impose a too large runtime overhead
        //     .iter()
        //     .find(|combi| !self.blocked_row_combinations.contains(combi)) // choose first block RowAddr-combination
        //     .expect("No SRA for nr-rows={nr_rows}");
        //
        // let activated_rows = ARCHITECTURE.precomputed_simultaneous_row_activations.get(row_combi).unwrap();
        // let ref_rows: Vec<RowAddress> = activated_rows.iter()
        //     .map(|row| row & (subarrayid_to_subarray_address(ref_subarray))) // make activated rows refer to the right subarray
        //     .collect();
        // let comp_rows: Vec<RowAddress>  = activated_rows.iter()
        //     .map(|row| row & (subarrayid_to_subarray_address(compute_subarray))) // make activated rows refer to the right subarray
        //     .collect();
        //
        //
        // // 1. Initialize rows in ref-subarray (if executing AND/OR)
        // // - TODO: read nr of frac-ops to issue from compiler-settings
        // let mut instruction_init_ref_subarray  = self.init_reference_subarray(ref_rows.clone(), logic_op);
        // next_instructions.append(&mut instruction_init_ref_subarray);
        //
        // // 2. Place rows in the simultaneously activated rows in the compute subarray (init other rows with 0 for OR, 1 for AND and same value for NOT)
        // let mut instructions_init_comp_subarray = self.init_compute_subarray( activated_rows.clone(), src_operands, logic_op, NeighboringSubarrayRelPosition::get_relative_position(compute_subarray, ref_subarray));
        // next_instructions.append(&mut instructions_init_comp_subarray);
        //
        // // SKIPPED: 2.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
        // // - only necessary once we find optimization to not write values to safe-space but reuse them diectly
        // // 2.2.1 if yes: move data to other rows for performing this op
        //
        // // 3. Issue actual operation
        // let mut actual_op = match logic_op {
        //     LogicOp::NOT => vec!(Instruction::ApaNOT(row_combi.0, row_combi.1)),
        //     LogicOp::AND | LogicOp::OR => vec!(Instruction::ApaAndOr(row_combi.0, row_combi.1)),
        //     LogicOp::NAND | LogicOp::NOR => vec!(Instruction::ApaAndOr(row_combi.0, row_combi.1), Instruction::ApaNOT(row_combi.0, row_combi.1)), // TODO: or the othyer way around (1st NOT)?
        // };
        //
        // next_instructions.append(&mut actual_op);
        // for (&comp_row, &ref_row) in comp_rows.iter().zip(ref_rows.iter()) {
        //     self.comp_state.dram_state.insert(comp_row, RowState { is_compute_row: true, live_value: Some(*next_candidate), constant: None });
        //     self.comp_state.value_states.insert(*next_candidate, ValueState { is_computed: true, row_location: Some(comp_row) });
        //     // ref subarray holds negated value afterwarsd
        //     self.comp_state.dram_state.insert(ref_row, RowState { is_compute_row: true, live_value: Some(next_candidate.invert()), constant: None });
        //     self.comp_state.value_states.insert(next_candidate.invert(), ValueState { is_computed: true, row_location: Some(ref_row) });
        // }
        //
        // // 4. Copy result data from dst NEAREST to the sense-amps into a safe-space row and update `value_state`
        // // TODO LAST: possible improvement - error correction over all dst-rows (eg majority-vote for each bit, votes weighted by distance to sense-amps?)
        //
        // next_instructions
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
    /// `compute_rows` are reservered rows which solely exist for performing computations, see [`Compiler::compute_row_activations`]
    is_compute_row: bool,
    /// `None` if the value inside this row is currently not live
    live_value: Option<Signal>,
    /// Mostly 0s/1s (for initializing reference subarray), see [`CompilationState::constant_values`]
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
    /// For each row in the dram-module store its state (whether it's a compute row or if not whether/which value is stored inside it
    dram_state: HashMap<RowAddress, RowState>,
    /// Stores row in which an intermediate result (which is still to be used by future ops) is currently located (or whether it has been computed at all)
    value_states: HashMap<Signal, ValueState>,
    /// Stores row location of constant
    /// - REMINDER: some constants are stored in fixed rows (!in each subarray), eg 0s and 1s for initializing reference subarray
    constant_values: HashMap<usize, RowAddress>,
    /// List of candidates (ops ready to be issued) prioritized by some metric by which they are scheduled for execution
    /// - NOTE: calculate Nodes `SchedulingPrio` using
    candidates: PriorityQueue<Signal, SchedulingPrio>,
    /// For each Subarray store which rows are free (and hence can be used for storing values)
    free_rows_per_subarray: HashMap<SubarrayId, Vec<RowAddress>>,
}

impl CompilationState {
    pub fn new(dram_state: HashMap<RowAddress, RowState>) -> Self {
        Self {
            dram_state,
            value_states: HashMap::new(),
            constant_values: HashMap::new(),
            candidates: PriorityQueue::new(),
            free_rows_per_subarray: HashMap::new(),
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
        let (selected_subarray, _) = compiler.select_compute_and_ref_subarray(vec!(RowAddress(0b1_000_000_000), RowAddress(0b1_000_010_000), RowAddress(0b111_000_000_000), RowAddress(0b10_100_000_000),));
        assert_eq!(selected_subarray, 0b1_000_000_000);
    }

    #[ignore]
    fn test_program_validity() {
        // 1. test that no APA activates a safe-space row

        // ..
    }
}
