use super::{
    architecture::{PRADAArchitecture},
};
use crate::prada::{architecture::{RowAddress, ARCHITECTURE, ROWS_PER_SUBARRAY}, extraction::CompilingCost, program::{Instruction, Program}};
use eggmock::{Id, Mig, NetworkWithBackwardEdges, Node, Signal};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

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

pub struct CompilationState<'n, N> {
    /// For each row in the dram-module store its state (whether it's a compute row or if not whether/which value is stored inside it
    dram_state: HashMap<RowAddress, RowState>,
    /// For each subarray it stores the row in which the `Signal` is located
    /// Assumption: currently only uses a single subarray
    value_states: HashMap<Signal, RowAddress>,
    /// For each Subarray store which rows are free (and hence can be used for storing values)
    /// - for now we'll limit ourselves to a single subarray
    free_rows_per_subarray: Vec<RowAddress>,

    network: &'n N,
    program: Vec<Instruction>,
    /// contains all not yet computed network nodes that can be immediately computed (i.e. all
    /// inputs of the node are already computed)
    candidates: FxHashSet<(Id, Mig)>,

    outputs: FxHashSet<Id>,
    leftover_use_count: FxHashMap<Id, usize>,
}

/// Main function
/// - called with all initial candidates (=leaves) already placed in some rows
pub fn compile<'a>(
    architecture: &'a PRADAArchitecture,
    network: &impl NetworkWithBackwardEdges<Node = Mig>,
) -> Result<Program<'a>, &'static str> {

    // init candidates, dram_state etc.
    let mut state = CompilationState::new(architecture, network);

    // dbg!("{:?}", state.value_states.clone());

    while !state.candidates.is_empty() {
        // choose next candidate
        let (id, node, _, _, _) = state
            .candidates
            .iter()
            .copied()
            .map(|(id, node)| {
                let outputs = state.network.node_outputs(id).count();
                let output = state.network.outputs().any(|out| out.node_id() == id);
                let not_present = node
                    .inputs()
                    .iter()
                    .map(|signal| {
                        // TODO: check if this is legit
                        let present = state.value_states.contains_key(signal);
                        !present as u8
                    })
                    .sum::<u8>();
                (id, node, not_present, outputs, output)
            })
            .min_by_key(|(_, _, not_present, outputs, output)| (*not_present, *outputs, !output))
            .unwrap();

        // if state.outputs.contains(&id) {
        //     println!("Computing outputs...");
        //     for (output, signal) in network.outputs().enumerate() {
        //         if signal.node_id() != id {
        //             continue;
        //         }
        //         if signal.is_inverted() && !state.value_states.contains_key(&signal) {
        //             let backup_row = state.free_rows_per_subarray.pop().expect("No empty rows anymore");
        //             let orig_row = *state.value_states.get(&signal.invert()).unwrap_or_else(|| panic!("Non-inverted version of signal {signal:?} isnt present too"));
        //
        //             // for now let's save the original (non-inverted) value in another row - TODO:
        //             // check if signal is ever needed again and only then do this
        //             state.program.push(Instruction::AAPRowCopy(orig_row, backup_row)); // first save in backup_row
        //             state.program.push(Instruction::N(backup_row)); // then negate
        //             state.value_states.insert(signal, orig_row); // inv signal is now stored where non-inv sig was previously
        //             state.dram_state.insert(orig_row, RowState{ is_compute_row: false, live_value: Some(signal.invert()), constant: None});
        //             state.dram_state.insert(backup_row, RowState{ is_compute_row: false, live_value: Some(signal), constant: None});
        //
        //             state.compute(id, node, None);
        //         } else {
        //             state.compute(id, node, Some(RowAddress(output as u64)));
        //         }
        //     }
        //     state.outputs.remove(&id);
        //
        //     // free rows if node isn't needed for further computations
        //     let leftover_uses = *state.leftover_use_count(id);
        //     if leftover_uses == 1 {
        //         for signal in network.node(id).inputs() {
        //             let row= state.value_states.remove(signal);
        //             if let Some(row_addr) = row {
        //                 state.dram_state.remove(&row_addr);
        //                 state.free_rows_per_subarray.push(row_addr);
        //             }
        //         }
        //     }
        // } else {
        state.compute(id, node, None);
        // }
    }

    // print in which rows the outputs have been placed
    for output in network.outputs() {
        if let Some(row) = state.value_states.get(&output) {
            println!("Output {output:?} in row {row}");
        } else {
            // check if inverted signal of `output` is there
            let row = if let Some(&inv_sig_row) = state.value_states.get(&output.invert()) {
                if !network.outputs().any(|o| o == output.invert()) {
                    state.program.push(Instruction::N(inv_sig_row));
                    state.value_states.remove(&output);
                    state.value_states.insert(output.invert(), inv_sig_row);
                    state.dram_state.insert(inv_sig_row, RowState{ is_compute_row: false, live_value: Some(output.invert()), constant: None});
                    inv_sig_row
                } else {
                    // if inverted signal is also an output we can't just overwrite it and have to
                    // save it in a separate row
                    let free_row = state.free_rows_per_subarray.pop().expect("OOM");
                    state.program.push(Instruction::AAPRowCopy(inv_sig_row, free_row));
                    state.program.push(Instruction::N(free_row));
                    inv_sig_row
                }
            } else {
                panic!("Output {output:?} nor its inverse are in a row??");
            };
            println!("Output {output:?} in row {row}");
        }
    }

    // outputs that are directly derived from inputs will not be computed by the loop above
    // let's do that here
    // for (idx, output_sig) in network.outputs().enumerate() {
    //     if !state.outputs.contains(&output_sig.node_id()) {
    //         continue;
    //     }
    //     state
    //         .program
    //         .signal_copy(output_sig, RowAddress(idx as u64));
    // }

    let CompilingCost{ runtime,  energy_consumption} = state.program.iter().map(|instr| {
        match instr {
            Instruction::N(_) => CompilingCost { runtime: 35, energy_consumption: 100},
            Instruction::AAPTRA(_,_,_) => CompilingCost { runtime: 49, energy_consumption: 150},
            Instruction::AAPRowCopy(_,_) => CompilingCost { runtime: 100, energy_consumption: 50},
        }
    }).sum();

    // println!("{:?}", state.program);

    let program = Program { architecture: &ARCHITECTURE , instructions: state.program, runtime_estimate: runtime, energy_consumption_estimate: energy_consumption };
    Ok(program)
}

impl<'a, 'n, N: NetworkWithBackwardEdges<Node = Mig>> CompilationState<'n, N> {
    pub fn new(architecture: &'a PRADAArchitecture, network: &'n N) -> Self {
        let mut candidates = FxHashSet::default();
        // check all parents of leafs whether they have only leaf children, in which case they are
        // candidates
        for leaf in network.leafs() {
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
        let outputs = network.outputs().map(|sig| sig.node_id()).collect();

        let (dram_state, value_states, free_rows) = CompilationState::get_init_states(network, (0..ROWS_PER_SUBARRAY) .map(RowAddress::from) .collect());
        Self {
            dram_state,
            value_states,
            // initially all rows are free
            free_rows_per_subarray: free_rows,
            network,
            candidates,
            // start with empty program (no instructions inside)
            program: vec!(),
            outputs,
            leftover_use_count: FxHashMap::default(),
        }
    }

    pub fn get_init_states(ntk: &'n N, mut free_rows_per_subarray: Vec<RowAddress> ) ->  (HashMap<RowAddress, RowState>, HashMap<Signal, RowAddress>, Vec<RowAddress>) {
        let mut dram_state = HashMap::new();
        let mut value_states = HashMap::new();
        // 0. Place constants `True`&`False`
        let row_for_false = free_rows_per_subarray.pop().expect("No more free rows");
        let row_state = RowState{ is_compute_row: false, live_value: None, constant: Some(0) }; // False
        dram_state.insert(row_for_false, row_state);
        println!("Place 0s into {row_for_false}");
        let row_for_true = free_rows_per_subarray.pop().expect("No more free rows");
        let row_state = RowState{ is_compute_row: false, live_value: None, constant: Some(std::usize::MAX) }; // True
        dram_state.insert(row_for_true, row_state);
        println!("Place 1s into {row_for_true}");

        let leafs = ntk.leafs();
        for id in leafs {
            let node = ntk.node(id);
            let next_row = free_rows_per_subarray.pop().expect("No more free rows");
            match node {
                Mig::Input(i) => {
                    println!("Input {id:?} placed in row {next_row}");
                    // TODO: check whether inverted or non-inverted version is needed and place only
                    // that one
                    let row_state = RowState{ is_compute_row: false, live_value: Some(Signal::new(id, false)), constant: None };
                    value_states.insert(Signal::new(id, false), next_row);
                    dram_state.insert(next_row, row_state);

                    let next_row = free_rows_per_subarray.pop().expect("No more free rows");
                    let row_state = RowState{ is_compute_row: false, live_value: Some(Signal::new(id, true)), constant: None };
                    value_states.insert(Signal::new(id, true), next_row);
                    dram_state.insert(next_row, row_state);
                }
                Mig::False => {
                    let row_state = RowState{ is_compute_row: false, live_value: Some(Signal::new(id, false)), constant: Some(0) };
                    value_states.insert(Signal::new(id, false), row_for_false);
                    dram_state.insert(row_for_false, row_state);

                    let row_state = RowState{ is_compute_row: false, live_value: Some(Signal::new(id, true)), constant: Some(std::usize::MAX) };
                    value_states.insert(Signal::new(id, true), row_for_true);
                    dram_state.insert(row_for_true, row_state);
                }
                _ => unreachable!("leaf node should be either an input or a constant"),
            };
        }

        (dram_state, value_states, free_rows_per_subarray)
    }

    pub fn leftover_use_count(&mut self, id: Id) -> &mut usize {
        self.leftover_use_count.entry(id).or_insert_with(|| {
            // or if node hasn't been touched yet: init `leftover_use_count` with nr uses
            self.network.node_outputs(id).count() + self.outputs.contains(&id) as usize
        })
    }

    /// Actually compute the operation behind `node` and store it into `out_address`
    pub fn compute(&mut self, id: Id, node: Mig, out_address: Option<RowAddress>) {
        // dbg!("Computing {:?}", id);
        // dbg!("Candidates: {:?}", &self.candidates);
        if !self.candidates.remove(&(id, node)) {
            panic!("not a candidate");
        }
        let Mig::Maj(signals) = node else {
            panic!("can only compute majs")
        };

        // get row addresses of require input operands
        let row_addresses: Vec<RowAddress> = signals.iter().map(|signal|
            if !self.value_states.contains_key(signal) {
                if let Some(row_inv_sig) = self.value_states.get(&signal.invert()) {
                    // if signal isn't there, first create it using the inverted signal
                    let free_row = self.free_rows_per_subarray.pop().expect("OOM");
                    self.program.push(Instruction::AAPRowCopy(*row_inv_sig, free_row));
                    self.program.push(Instruction::N(free_row));

                    self.value_states.insert(*signal, free_row);
                    self.dram_state.insert(free_row, RowState { is_compute_row: false, live_value: Some(*signal), constant: None});
                    free_row
                } else {
                    panic!("Input Signal {signal:?} nor its inverted version are present. Why is {id:?} a candidate then?");
                }
            } else {
                *self.value_states.get(signal).unwrap()
            }
        ).collect();


        // update `leftover_use_count` of parent of this signal & free row if operand is not needed
        // anymore
        if ! self.network.outputs().any(|x| x.node_id() == id) {
            self.network.node_outputs(id).for_each(|parent| {
                self.leftover_use_count.entry(parent).and_modify(|v| *v -= 1);
                if self.leftover_use_count.get(&parent) == Some(&1) {
                    if let Some(row) = self.value_states.remove(&Signal::new(parent, false)) {
                        self.dram_state.remove(&row);
                        self.free_rows_per_subarray.push(row);
                    }
                    if let Some(row) = self.value_states.remove(&Signal::new(parent, true)) {
                        self.dram_state.remove(&row);
                        self.free_rows_per_subarray.push(row);
                    }
                }
            });
        }

        // move values into safe rows if they're needed in future (=still live)
        for signal in signals {
            if *self.leftover_use_count(signal.node_id()) > 1  {
                let next_free_row = self.free_rows_per_subarray.pop().expect("OOM");
                let row_addr = *self.value_states.get(&signal).unwrap_or_else(|| panic!("Input Signal with node-id={:?} not present. Why is {id:?} a candidate then?", signal.node_id()));
                self.value_states.insert(signal, next_free_row);
                self.dram_state.insert(next_free_row, RowState { is_compute_row: false, live_value: Some(signal), constant: None});
                self.program.push(Instruction::AAPRowCopy(row_addr, next_free_row));
            }
        }

        // perform MAJ3
        self.program.push(Instruction::AAPTRA(row_addresses[0], row_addresses[1], row_addresses[2]));
        self.value_states.insert(Signal::new(id, false), row_addresses[0]);
        self.dram_state.insert(row_addresses[0], RowState { is_compute_row: false, live_value: Some(Signal::new(id, false)), constant: None } );
        // keep result only in one of the addresses, free the remaining rows
        self.dram_state.remove(&row_addresses[1]);
        self.dram_state.remove(&row_addresses[2]);
        self.free_rows_per_subarray.append(&mut vec!(row_addresses[1], row_addresses[2]));

        // lastly, determine new candidates
        for parent_id in self.network.node_outputs(id) {
            let parent_node = self.network.node(parent_id);
            if parent_node
                .inputs()
                .iter()
                .all(|s| self.value_states.contains_key(s) || self.value_states.contains_key(&s.invert()))
            {
                self.candidates.insert((parent_id, parent_node));
            }
        }
    }
}
