//! # Literature
//!
//! - [1] Functionally-Complete Boolean Logic in Real DRAM Chips: Experimental Characterization and Analysis, 2024
//! - [2] FracDRAM: Fractional Values in Off-the-Shelf DRAM, 2022
//! - [3] PULSAR: Simultaneous Many-Row Activation for Reliable and High-Performance Computing in Off-the-Shelf DRAM Chips, 2024
//!
//! # Submodules
//!
//! - [`architecture`] - defines Instructions (and performance-metrics of Instructions in that architecture) used in FC-DRAM
//! - [`compilation`] - compiles given LogicNetwork for FC-DRAM architecture
//! - [`generator`] — Generates output code or reports based on analysis.
mod compilation;
mod extraction;
mod optimization;
mod program;
mod architecture;

use std::sync::LazyLock;
use std::time::Instant;

use self::compilation::compile;
use self::extraction::CompilingCostFunction;

use eggmock::egg::{rewrite, EGraph, Extractor, Id, Rewrite, Runner};
use eggmock::{
    Aig, AigLanguage, AigReceiverFFI, Provider, Receiver, ReceiverFFI, Rewriter,
    RewriterFFI, // TODO: add AOIG-rewrite (bc FC-DRAM supports AND&OR natively)?
};
use program::*;
use architecture::*;

/// Rewrite rules to use in E-Graph Rewriting (see [egg](https://egraphs-good.github.io/))
/// TODO: adjust rewriting rules to FCDRAM (=AND/OR related rewrites like De-Morgan?)
static REWRITE_RULES: LazyLock<Vec<Rewrite<AigLanguage, ()>>> = LazyLock::new(|| {
    let mut rules = vec![
        // TODO: add "or"
        rewrite!("commute-and"; "(and ?a ?b)" => "(and ?b ?a)"),
        rewrite!("and-1"; "(and ?a 1)" => "?a"),
        rewrite!("and-0"; "(and ?a 0)" => "0"),
        rewrite!("not_not"; "(! (! ?a))" => "?a"),
        // rewrite!("maj_1"; "(maj ?a ?a ?b)" => "?a"),
        // rewrite!("maj_2"; "(maj ?a (! ?a) ?b)" => "?b"),
        // rewrite!("associativity"; "(maj ?a ?b (maj ?c ?b ?d))" => "(maj ?d ?b (maj ?c ?b ?a))"),
    ];
    // rules.extend(rewrite!("invert"; "(! (maj ?a ?b ?c))" <=> "(maj (! ?a) (! ?b) (! ?c))"));
    // rules.extend(rewrite!("distributivity"; "(maj ?a ?b (maj ?c ?d ?e))" <=> "(maj (maj ?a ?b ?c) (maj ?a ?b ?d) ?e)"));
    rules
});

/// Store compilation output and timing statistics how long compilation stages took
/// TODO: unit of t? sec or ms?
struct CompilingReceiverResult {
    /// Actual compilation result
    output: CompilerOutput,

    /// Statistics about compilation
    t_runner: u128,
    t_extractor: u128,
    t_compiler: u128,
}

/// Compilation result (program + E-Graph)
#[ouroboros::self_referencing]
struct CompilerOutput {
    /// Result E-Graph
    graph: EGraph<AigLanguage, ()>,
    /// (, output-nodes)
    #[borrows(graph)]
    #[covariant]
    ntk: (
        Extractor<'this, CompilingCostFunction, AigLanguage, ()>,
        Vec<Id>,
    ),
    /// Compiled Program
    #[borrows(ntk)]
    program: Program,
}

/// Initiates compilation and prints compilation-statistics (and program if `settings.verbose=true`
/// - returned receiver allows converting result-graph in both directions (C++ <=> Rust)
/// - `settings`: compiler-options
fn compiling_receiver<'a>(
    rules: &'a [Rewrite<AigLanguage, ()>],
    settings: CompilerSettings,
) -> impl Receiver<Result = CompilingReceiverResult, Node = Aig> + use<'a> {
    // REMINDER: EGraph implements `Receiver`
    EGraph::<AigLanguage, _>::new(()).map(move |(graph, outputs)| { // `.map()` of `Provider`-trait:
        let t_runner = std::time::Instant::now();

        // 1. Create E-Graph: run equivalence saturation
        let runner = Runner::default().with_egraph(graph).run(rules);

        let t_runner = t_runner.elapsed().as_millis();

        if settings.verbose {
            println!("== Runner Report");
            runner.print_report();
        }

        let graph = runner.egraph;

        let mut t_extractor = 0;
        let mut t_compiler = 0;

        // 2. Given E-Graph: Compile the actual program
        let output = CompilerOutput::new(
            graph,
            |graph| {
                let start_time = Instant::now();
                // TODO: what is the extractor for??
                let extractor = Extractor::new(
                    graph,
                    CompilingCostFunction {}, // TODO: provide CostFunction !!
                );
                t_extractor = start_time.elapsed().as_millis();
                (extractor, outputs)
            },
            |ntk| {
                let start_time = Instant::now();

                // ===== MAIN CALL =====
                let program = compile(&ntk.with_backward_edges()); // actual compilation !!
                // =====================

                t_compiler = start_time.elapsed().as_millis();
                // print program if compiler-setting is set
                if settings.print_program || settings.verbose {
                    if settings.verbose {
                        println!("== Program")
                    }
                    println!("{program}");
                }
                program
            },
        );
        if settings.verbose {
            println!("== Timings");
            println!("t_runner: {t_runner}ms");
            println!("t_extractor: {t_extractor}ms");
            println!("t_compiler: {t_compiler}ms");
        }
        CompilingReceiverResult {
            output,
            t_runner,
            t_extractor,
            t_compiler,
        }
    })
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
/// Compiler options
/// - TODO: add flags like minimal success-rate for program
struct CompilerSettings {
    /// Whether to print the compiled program
    print_program: bool,
    /// Whether to enable verbose output
    verbose: bool,
    /// Minimal success rate to be guaranteed for success compiled program
    /// REMINDER: FCDRAM-operations dont have a 100%-success rate to create the correct results
    min_success_rate: u64,
}

struct FCDramRewriter(CompilerSettings);

impl Rewriter for FCDramRewriter {
    type Node = Aig;
    type Intermediate = CompilingReceiverResult;

    fn create_receiver(
        &mut self,
    ) -> impl Receiver<Node = Aig, Result = CompilingReceiverResult> + 'static {
        compiling_receiver(REWRITE_RULES.as_slice(), self.0)
    }

    fn rewrite(
        self,
        result: CompilingReceiverResult,
        output: impl Receiver<Node = Aig, Result = ()>,
    ) {
        result.output.borrow_ntk().send(output);
    }
}

/// ??
#[no_mangle]
extern "C" fn fcdram_rewriter(settings: CompilerSettings) -> AigReceiverFFI<RewriterFFI<Aig>> {
    RewriterFFI::new(FCDramRewriter(settings))
}

/// Statistic results about Compilation-Process
#[repr(C)]
struct CompilerStatistics {
    egraph_classes: u64,
    egraph_nodes: u64,
    egraph_size: u64,

    instruction_count: u64,

    t_runner: u64,
    t_extractor: u64,
    t_compiler: u64,
}

/// Main function called from `.cpp()` file - receives compiler settings
#[no_mangle]
extern "C" fn fcdram_compile(settings: CompilerSettings) -> AigReceiverFFI<CompilerStatistics> {
    // todo!()
    // TODO: create example `ARCHITECTURE` implementing `FCDRAMArchitecture`
    let receiver =
        compiling_receiver(REWRITE_RULES.as_slice(), settings).map(|res| {
            let graph = res.output.borrow_graph();
            CompilerStatistics {
                egraph_classes: graph.number_of_classes() as u64,
                egraph_nodes: graph.total_number_of_nodes() as u64,
                egraph_size: graph.total_size() as u64,
                instruction_count: res.output.borrow_program().instructions.len() as u64,
                t_runner: res.t_runner as u64,
                t_extractor: res.t_extractor as u64,
                t_compiler: res.t_compiler as u64,
            }
        });
    AigReceiverFFI::new(receiver)
}
