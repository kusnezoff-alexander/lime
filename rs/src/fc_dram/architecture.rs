//! Contains all architecture-specific descriptions
//! - [`FCDRAMArchitecture`] = DRAM-module-specific specific implementation of FCDRAMArchitecture
//! - [`Instruction`] = contains all instructions supported by FC-DRAM architecture
//! - [ ] `RowAddress`: utility functions to get subarray-id and row-addr within that subarray from
//!
//! RowAddress (eg via bit-shifting given bitmasks for subarray-id & row-addr to put on-top of RowAddress

use std::{collections::{HashMap, HashSet}, fmt::{Display, Formatter}, sync::LazyLock};

use log::debug;

/// Main variable specifying architecture of DRAM-module for which to compile for
/// - this is currently just an example implementation for testing purpose; (TODO: make this configurable at runtime)
///
/// TODO: add field to simulate row-decoder circuitry, needed for impl Simultaneous-row-activation
/// TODO: make this configurable at runtime
pub static ARCHITECTURE: LazyLock<FCDRAMArchitecture> = LazyLock::new(|| {
    const NR_SUBARRAYS: i64 = 2i64.pow(7);
    const ROWS_PER_SUBARRAY: i64 = 2i64.pow(9);

    // Implementation of the Hypothetical Row Decoder from [3] Chap4.2
    // - GWLD (Global Wordline Decoder)=decode higher bits to select addressed subarray
    // - LWLD (Local Wordline Decoder)=hierarchy of decoders which decode lower bits; latches remain set when using `APA`
    //      - see [3] Chap4.2: nr of Predecoders in LWLD determines number & addresses of simultaneously activated rows
    //  - does work for the example shown in [3] Chap3.2: `APA(256,287)` activates rows `287,286,281,280,263,262,257,256`
    // TODO: add overlapping of higher-order-bits (GWLD)
    // TODO: init architecture a run-time, eg from config file
    // TODO: maybe evaluate statically?
    let get_activated_rows_from_apa = |row1: RowAddress, row2: RowAddress| -> Vec<RowAddress> {
        // 1. Define Predecoders by defining for which of the bits they're responsible
        // each Predecoder is resonsible for some of the lower order bits
        let predecoder_bitmasks = [
            0b110000000, // first predecoder (PE) predecodes bits[8,7]
            0b001100000, // Predecoder PD
            0b000011000, // Predecoder PC
            0b000000110, // Predecoder PB
            0b000000001, // last predecoder (PA) predecodes bits[0]
        ];

        // for each predecoder store which bits will remain set due to `APA(row1,row)`:
        let overlapping_bits = [
            // latches set by `ACT(row1)`  --- latches set by `ACT(row2)`
            [ row1 & predecoder_bitmasks[0], row2 & predecoder_bitmasks[0]],
            [ row1 & predecoder_bitmasks[1], row2 & predecoder_bitmasks[1]],
            [ row1 & predecoder_bitmasks[2], row2 & predecoder_bitmasks[2]],
            [ row1 & predecoder_bitmasks[3], row2 & predecoder_bitmasks[3]],
            [ row1 & predecoder_bitmasks[4], row2 & predecoder_bitmasks[4]],
        ];

        let mut activated_rows  = vec!();       // TODO: get other activated rows and add them to `activated_rows`
        // compute all simultaneously activated rows
        for i in 0..1 << predecoder_bitmasks.len() {
            let activated_row = overlapping_bits.iter()
                // start with all row-address bits unset (=0) and first predecoder stage (=1)
                .fold((0 as RowAddress, 1), |(row, predecoder_stage_onehot), new_row_bits|{
                    let bitmask_to_choose = (i & predecoder_stage_onehot) > 0;
                    (row | new_row_bits[bitmask_to_choose as usize], predecoder_stage_onehot << 1)
                });
            activated_rows.push(activated_row.0);
        }
        // debug!("`APA({row1},{row2})` activates the following rows simultaneously: {activated_rows:?}");
        activated_rows.dedup(); // no need for `.unique()` since this implementation adds equivalent RowAddresses one after the other (!check!!)
                                 // NOTE: works in-place

        // remove duplicate entries
        activated_rows.into_iter().collect::<HashSet<_>>().into_iter().collect()
    };

    let get_distance_of_row_to_sense_amps = |row: RowAddress| -> RowDistanceToSenseAmps {
        // ASSUMPTION: last & first rows only have sense-amps from one side
        // TODO: is this true? or do all subarrays have a line of sense-amps on both of their ends??
        let distance_to_nearest_sense_amp_in_nr_rows = if row < ROWS_PER_SUBARRAY {
            // this row is in the first subarray
            row // row-addr = distance to nearest sense-amps
        } else if  row > (NR_SUBARRAYS-1)*ROWS_PER_SUBARRAY {
            // this row is in the last subarray
            row - (NR_SUBARRAYS-1)*ROWS_PER_SUBARRAY // =distance to above sense-amps
        } else {
            // let subarray_id = row / rows_per_subarray;
            let row_nr_in_subarray = row % ROWS_PER_SUBARRAY;
            if row_nr_in_subarray < ROWS_PER_SUBARRAY {
                // row is in the 1st half of the subarray and hence nearer to the "previous" sense-amps
                row_nr_in_subarray
            } else {
                // row is in the 2nd half of the subarray and hence nearer to the "previous" sense-amps
                row_nr_in_subarray - ROWS_PER_SUBARRAY/2
            }

        };
        match distance_to_nearest_sense_amp_in_nr_rows {
            i if i < ROWS_PER_SUBARRAY / 2 / 3 => RowDistanceToSenseAmps::Close,   // 1st third  of subarray-half
            i if i < ROWS_PER_SUBARRAY / 2 / 6 => RowDistanceToSenseAmps::Middle,  // 2nd third  of subarray-half
            _ => RowDistanceToSenseAmps::Far,                                           // everything else is treated as being far away
        }
    };

    let mut precomputed_simultaneous_row_activations = HashMap::new();
    for i in 0..ROWS_PER_SUBARRAY {
        precomputed_simultaneous_row_activations.insert((i,i), vec!(i)); // special case: no other row is activated when executing `APA(r1,r1)`
        for j in i+1..ROWS_PER_SUBARRAY {
            let activated_rows = get_activated_rows_from_apa(i, j);
            precomputed_simultaneous_row_activations.insert((i,j), activated_rows.clone());
            precomputed_simultaneous_row_activations.insert((j,i), activated_rows);
        }
    }
    debug!("Precomputed SRAs: {:#?}", precomputed_simultaneous_row_activations.iter().take(20).collect::<Vec<_>>());

    let precomputed_activated_rows_nr_to_row_address_tuple_mapping=  precomputed_simultaneous_row_activations.iter().fold(HashMap::new(), |mut acc: HashMap<u8, Vec<(RowAddress,RowAddress)>>, (key, vec)| {
            acc.entry(vec.len() as u8).or_default().push(*key);
            acc
    });
    debug!("SRAs row-nr to row-addr mapping: {:#?}", precomputed_activated_rows_nr_to_row_address_tuple_mapping.iter().map(|(k,v)| format!("{k} rows activated in {} addr-combinations", v.len())).collect::<Vec<String>>());

    FCDRAMArchitecture {
        nr_subarrays: NR_SUBARRAYS,
        rows_per_subarray: ROWS_PER_SUBARRAY,
        get_activated_rows_from_apa,
        precomputed_simultaneous_row_activations,
        precomputed_activated_rows_nr_to_row_address_tuple_mapping,
        get_distance_of_row_to_sense_amps,
    }
});

/// - ! must be smaller than `rows_per_subarray * nr_subarrays` (this is NOT checked!)
pub type RowAddress = i64;

pub struct FCDRAMArchitecture {
    /// Nr of subarrays in a DRAM module
    pub nr_subarrays: i64,
    /// Nr of rows in a single subarray
    pub rows_per_subarray: i64,
    /// Returns all activated rows when issuing `APA(row1, row2)`
    /// - NOTE: `row1`,`row2` are expected to reside in adjacent subarrays
    /// - NOTE: the simultaneously activated rows are expected to have the same addresses in both subarrays
    ///     - eg `APA(11,29)` (with 1st digit=subarray-id, 2nd digit=row-id) could simultaneously activate rows `0,1,7,9` in subarray1 and subarray2
    get_activated_rows_from_apa: fn(RowAddress, RowAddress) -> Vec<RowAddress>,
    /// Stores which rows are simultaneously activated for each combination of Row-Addresses (provided to `APA`-operation)
    /// - REASON: getting the simultaneously activated will probably be requested very frequently (time-space tradeoff, rather than recomputing on every request))
    pub precomputed_simultaneous_row_activations: HashMap<(RowAddress, RowAddress), Vec<RowAddress>>,
    /// For each nr of activated rows get which tuple of row-addresses activate the given nr of rows
    /// - use to eg restrict the choice of row-addresses for n-ary AND/OR (eg 4-ary AND -> at least activate 8 rows; more rows could be activated when using input replication)
    pub precomputed_activated_rows_nr_to_row_address_tuple_mapping: HashMap<u8, Vec<(RowAddress,RowAddress)>>,
    // TODO: params for calculating distance btw row and sense-amp, ... (particularly where
    // sense-amps are placed within the DRAM module ?!
    /// Given a row-addr this returns the distance of it to the sense-amps (!determinse
    /// success-rate of op using that `row` as an operand) (see [1] Chap5.2)
    /// - NOTE: Methodology used in [1] to determine distance: RowHammer
    pub get_distance_of_row_to_sense_amps: fn(RowAddress) -> RowDistanceToSenseAmps,
}

/// Implement this trait for your specific DRAM-module to support FCDRAM-functionality
/// - contains the mapping of logical-ops to FCDRAM-Architecture (see
/// [`FCDRAMArchitecture::get_instructions_implementation_of_logic_ops`]
///
/// # Possible Changes in Future
///
/// - add trait-bound to a more general `Architecture`-trait to fit in the overall framework?
impl FCDRAMArchitecture {

    /// Returns FC-DRAM operations to perform for each logical operation, with operand-rows NOT set !!!
    /// - addresses of row operands need to be overwritten during compilation !
    ///
    /// REMINDER: for OR&AND additional [`Instruction::FracOp`]s need to be issued to setup the
    /// reference subarray containing `reference_rows` in order to perform the given `logic_op` on
    /// the `compute_rows` inside the computation rows
    ///
    /// REMINDER: do increase the success rate of `FracOp` storing a fractional value (`V_{DD}/2`
    /// in this case, several FracOps are usually issued)
    /// - ->`FracOp`s are replicated during compilation as necessary, this is done during compilation
    ///
    /// NOTE: `compute_rows` are expected to lay in the same subarray and `reference_rows` in one
    /// subarray adjacent to the compute subarray (!this is not checked but assumed to be true!)
    fn get_instructions_implementation_of_logic_ops(logic_op: SupportedLogicOps) -> Vec<Instruction> {
        match logic_op {
            SupportedLogicOps::NOT => vec!(Instruction::APA(-1, -1)),
            SupportedLogicOps::AND => vec!(Instruction::FracOp(-1), Instruction::APA(-1, -1)),
            SupportedLogicOps::OR => vec!(Instruction::FracOp(-1), Instruction::APA(-1, -1)),
            SupportedLogicOps::NAND => {
                // 1. AND, 2. NOT
                FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(SupportedLogicOps::AND)
                    .into_iter()
                    .chain( FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(SupportedLogicOps::NOT))
                    .collect()
            },
            SupportedLogicOps::NOR => {
                // 1. OR, 2. NOT
                FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(SupportedLogicOps::OR)
                    .into_iter()
                    .chain( FCDRAMArchitecture::get_instructions_implementation_of_logic_ops(SupportedLogicOps::NOT))
                    .collect()
            }
        }
    }
}

/// Categories of distances of rows to sense-amops
pub enum RowDistanceToSenseAmps {
    Close,
    Middle,
    Far,
}

/// Instructions used in FC-DRAM
/// - NOT: implemented using `APA`
/// - AND/OR: implemented by (see [1] Chap6.1.2)
///     1. setting `V_{AND}`/`V_{OR}` in reference subarray and then issuing (using FracOperation
///        for storing `V_{DD}/2`)
///     2. Issue `APA(R_{REF},R_{COM})` to simultaneously activate `N` rows in reference subarray
///        and `N` rows in compute subarray
///     3. Wait for `t_{RAS}` (=overwrites activated cells in compute subarray with AND/OR-result)
///     4. Issue `PRE` to complete the operation
///
/// Additionally RowClone-operations are added for moving data around if needed (eg if valid data
/// would be affected by following Simultaneous-Row-Activations)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Needed for initializing neutral row in reference subarray (to set `V_{AND}`/`V_{OR}` (see
    /// [1](
    /// Implemented using AP without any extra cycles in between) (see [2])
    /// - `PRE` "interrupt the process of row activation, and prevent the sense amplifier from being enabled"
    FracOp(RowAddress),
    /// Multiple-Row Activation: `ACT R_F -> PRE -> ACT R_L -> PRE` of rows `R_F`,`R_L` for rows within
    /// different subarrays. As a result `R_L` holds the negated value of `R_F` (see Chap5.1 of
    /// PaperFunctionally Complete DRAMs
    /// Used to implement NOT directly
    APA(RowAddress,RowAddress), // TODO: Rename to SimultaneousRowActivation or sth the like ?
    /// Fast-Parallel-Mode RowClone for cloning row-data within same subarray
    /// - corresponds to `AA`, basically copies from src-row -> row-buffer -> dst-row
    /// - first operand=src, 2nd operand=dst where `src` and `dst` MUST reside in the same subarray !
    RowCloneFPM(RowAddress, RowAddress),
    /// Copies data from src (1st operand) to dst (2nd operand) using RowClonePSM, which copies the
    /// data from `this_bank(src_row) -> other_bank(rowX) -> this_bank(dst_row)` (where
    /// `other_bank` might be any other bank). Since this copy uses the internal DRAM-bus it works
    /// on cacheline-granularity (64B) which might take some time for 8KiB rows...
    /// - see [4] Chap3.3 for `TRANSFER`-instruction
    RowClonePSM(RowAddress, RowAddress),
}

impl Display for Instruction {
 fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let description = match self {
            Instruction::FracOp(row) => format!("AP({row})"),
            Instruction::APA(row1,row2) => format!("APA({row1},{row2})"),
            Instruction::RowCloneFPM(row1,row2) => format!("AA({row1},{row2})"),
            Instruction::RowClonePSM(row1,row2) => format!("
                TRANSFER(<this_bank>{row1},<other_bank>(rowX))
                TANSFER(<other_bank>rowX,<this_bank>{row2})
            "),
        };
        write!(f, "{}", description)
    }
}

/// TODO: where to put logic for determining which rows are activated simultaneously given two
/// row-addresses
impl Instruction {
    /// Return Addreses of Rows which are used by this instruction (=operand-rows AND result-row)
    /// - REMINDER: although only two row-operands are given to `APA`, more rows can be/are affected due to *Simultaneous Row Activation* (see [3])
    ///
    /// TODO
    pub fn used_addresses(
        &self,
    ) -> Vec <RowAddress> {
    // ) -> impl Iterator<Item = RowAddress> {
        todo!()
        // let from = match self {
        //     Instruction::AAP(from, _) => from,
        //     Instruction::AP(op) => op,
        // }
        // .row_addresses(architecture);
        // let to = match self {
        //     Instruction::AAP(_, to) => Some(*to),
        //     _ => None,
        // }
        // .into_iter()
        // .flat_map(|addr| addr.row_addresses(architecture));
        // from.chain(to)
    }

    /// Returns all row-addresses whose values are overriden by this instruction
    /// TODO
    pub fn overridden_rows<'a>(
        &self,
    ) -> Vec <RowAddress> {
    // ) -> impl Iterator<Item = RowAddress> {
        todo!()
    }

    pub fn get_nr_memcycles(&self) -> u16 {
        match self {
            Instruction::FracOp(__) => 7,     // see [2] ChapIII.A, (two cmd-cycles + five idle cycles)
            Instruction::APA(_, _) => 3,            // NOTE: this is not explicitly written in the paper, TODO: check with authors
            Instruction::RowCloneFPM(_, _) => 2,    // see [4] Chap3.2
            Instruction::RowClonePSM(_, _) => 256,  // =(8192B/64B)*2 (*2 since copies two time, to and from `<other_bank>` on 64B-granularity
        }
    }
}

/// Contains logical operations which are supported (natively) on FCDRAM-Architecture
/// - see [`FCDRAMArchitecture::get_instructions_implementation_of_logic_ops`] for how these
/// logic-ops are mapped to FCDRAM-instructions
pub enum SupportedLogicOps {
    NOT,
    AND,
    OR,
    /// implemented using AND+NOT
    NAND,
    /// implemented using OR+NOT
    NOR,
}

/// Implements behavior of the RowDecoderCircuitry as described in [3]
/// TODO: remove in favor of passing arbitrary closure to [`FCDRAMArchitecture::get_activated_rows_from_apa`]
pub trait RowDecoder {
    /// Returns vector of simultaneously activated rows when issuing `APA(r1,r2)`-cmd
    /// NOTE: this may depend on the used DRAM - see [3] for a method for reverse-engineering
    /// which rows are activated simultaneously (also see RowClone)
    fn get_simultaneously_activated_rows_of_apa_op(&self, r1: RowAddress, r2: RowAddress) -> Vec<RowAddress>;

    // TODO: get activation pattern for given rows r1,r2 (N:N vs N:2N) - or just check whether
    // N:2N: is supported and let `get_simultaneously_activated_rows_of_apa_op()` handle the rest?
}

