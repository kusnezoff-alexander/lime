//!
//! - [`compile()`] = main function - compiles given logic network for the given [`architecture`]
//! into a [`program`] using some [`optimization`]

use super::{
    architecture::{FCDRAMArchitecture, Instruction, ARCHITECTURE}, optimization::optimize, Program, ProgramState, RowAddress
};
use eggmock::{Aig, ComputedNetworkWithBackwardEdges, Id, NetworkWithBackwardEdges, Node, Signal};
use log::debug;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;


/// Places given signal onto rows (modifying `comp_state`)
/// - NOTE: `signals` are assumed to be activated together during execution and are hence placed s.t. minimal nr of other rows holding valid data are simultaneously activated
/// - [ ] make sure that operation to be executed on those rows won't simultaneously activate other rows holding valid data which will be used by future operations
pub fn place_signals_onto_rows(
    comp_state: &mut impl NetworkWithBackwardEdges<Node = Aig>, signals: Vec<Id>
) {

}

/// Compiles given `network` intto a FCDRAM-[`Program`] that can be run on given `architecture`
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
    network: &impl NetworkWithBackwardEdges<Node = Aig>,
) -> Program {

    // debug!("Compiling {:?}", network);
    // 0. Setup: store all network-nodes yet to be compiled
    let comp_state = CompilationState::new(network); // initializes `.candidates()` with inputs + nodes whose src-operands are all inputs
    let program = Program::new(vec!());

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

    while !comp_state.candidates.is_empty() {
        // 1. Make sure rows are placed appropriately (starting from inputs)

        // 1.1 Determine in which rows src-operands for the next candidate-op are located

        // 1.2 Check if issuing `APA(src1,src2)` would activate other rows which hold valid data
            // 1.2.1 if yes: move data to other rows for performing this op

        // 1.3 Prepare performing the actual op (setup reference subarray)
            // 1.3.1 If activated rows in reference subarray holds valid data: spill to other rows

        // 1.4 Issue actual operation
    }

    // let (outputs, leaves) = (network.outputs(), network.leaves());

    // Program {
    //     instructions: vec!(Instruction::FracOp(-1)) ,
    // }
    // todo!()
    // optimize(&mut program);
    program
}

pub struct Compiler {
    // comp_state: CompilationState<'n, N>,
}

impl Compiler {
    fn new() -> Self {
        Compiler{}
    }

    fn compile() -> Program {
        Program {
            instructions: vec!(),
            input_row_operands_placementl: HashMap::new(),
            output_row_operands_placementl: HashMap::new(),
        }
    }

    /// Place inputs onto appropriate rows
    /// TODO: algo which looks ahead which input-row-placement might be optimal (->reduce nr of
    /// move-ops to move intermediate results around & keep inputs close to sense-amps
    fn init_rows_with_inputs() {

    }
}

/// Keep track of current progress of the compilation (eg which rows are used, into which rows data is placed, ...)
pub struct CompilationState<'n, N> {
    /// Logic-Network which is to be compiled
    network: &'n N,
    /// Signals whose inputs have already been calculated and are currently placed in some row in
    /// the DRAM module (see [`Self::signal_to_row_mapping`])
    candidates: FxHashSet<(Id, Aig)>,

    /// Stores which intermediate results are stored at which row-addresses
    /// - ✓ use to determine at which row-addresses output-data can be found
    signal_to_row_mapping: HashMap<Signal, RowAddress>,
    /// Rows which are free (don't hold data which is still needed)
    free_rows: Vec<RowAddress>,
    ///
    row_states: [i32; ARCHITECTURE.nr_subarrays as usize ]
}

impl<'n, P: NetworkWithBackwardEdges<Node = Aig>> CompilationState<'n, P> {
    /// initializes `self..candidates()` with inputs + nodes whose src-operands are all inputs
    pub fn new(network: &'n P) -> Self {
        let mut candidates = FxHashSet::default();
        // check all parents of leaves whether they have only leaf children, in which case they are
        // candidates (since all of their inputs are calculated then)
        for leaf in network.leaves() { // =inputs
            for candidate_id in network.node_outputs(leaf) {
                let candidate = network.node(candidate_id);
                if candidate
                    .inputs()
                    .iter()
                    .all(|signal| network.node(signal.node_id()).is_leaf())
                {
                    candidates.insert((candidate_id, candidate));
                }
            }
        }

        // let outputs = network
        //     .outputs()
        //     .enumerate()
        //     .map(|(id, sig)| (sig.node_id(), (id as i64, sig)))
        //     .collect();

        let total_nr_rows_in_dram_module = ARCHITECTURE.nr_subarrays;
        Self {
            network,
            candidates,
            signal_to_row_mapping: HashMap::new(),
            free_rows: (0..=total_nr_rows_in_dram_module).collect(), // we start with all rows being free at the beginning
        }
    }
}
