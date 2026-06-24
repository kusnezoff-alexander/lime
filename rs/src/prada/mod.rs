mod architecture;
mod compilation;
mod extraction;
mod program;
mod rows;

use std::sync::LazyLock;
use std::time::Instant;

use self::compilation::compile;
use self::extraction::CompilingCostFunction;

use crate::opt_extractor::{OptExtractionNetwork, OptExtractor};
use crate::prada::architecture::{PRADAArchitecture, ARCHITECTURE};
use eggmock::egg::{rewrite, EGraph, Rewrite, Runner};
use eggmock::{Mig, MigLanguage, MigReceiverFFI, Network, Receiver, ReceiverFFI};
use program::*;
use rows::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitwiseOperand {
    T(u8),
}

impl BitwiseOperand {
    pub fn row(&self) -> BitwiseRow {
        match self {
            BitwiseOperand::T(t) => BitwiseRow::T(*t),
        }
    }
    pub fn inverted(&self) -> bool {
        todo!()
        // matches!(self, BitwiseOperand::DCC { inverted: true, .. })
    }
}


static REWRITE_RULES: LazyLock<Vec<Rewrite<MigLanguage, ()>>> = LazyLock::new(|| {
    let mut rules = vec![
        rewrite!("commute_1"; "(maj ?a ?b ?c)" => "(maj ?b ?a ?c)"),
        rewrite!("commute_2"; "(maj ?a ?b ?c)" => "(maj ?a ?c ?b)"),
        rewrite!("not_not"; "(! (! ?a))" => "?a"),
        rewrite!("maj_1"; "(maj ?a ?a ?b)" => "?a"),
        rewrite!("maj_2"; "(maj ?a (! ?a) ?b)" => "?b"),
        rewrite!("associativity"; "(maj ?a ?b (maj ?c ?b ?d))" => "(maj ?d ?b (maj ?c ?b ?a))"),
    ];
    rules.extend(rewrite!("invert"; "(! (maj ?a ?b ?c))" <=> "(maj (! ?a) (! ?b) (! ?c))"));
    rules.extend(rewrite!("distributivity"; "(maj ?a ?b (maj ?c ?d ?e))" <=> "(maj (maj ?a ?b ?c) (maj ?a ?b ?d) ?e)"));
    rules
});

struct CompilingReceiverResult<'a> {
    output: CompilerOutput<'a>,

    t_runner: u128,
    t_extractor: u128,
    t_compiler: u128,
}

#[ouroboros::self_referencing]
struct CompilerOutput<'a> {
    graph: EGraph<MigLanguage, ()>,
    #[borrows(graph)]
    #[covariant]
    ntk: OptExtractionNetwork<OptExtractor<'this, CompilingCostFunction<'a>, MigLanguage, ()>>,
    #[borrows(ntk)]
    program: Program<'a>,
}

fn compiling_receiver<'a>(
    architecture: &'a PRADAArchitecture,
    rules: &'a [Rewrite<MigLanguage, ()>],
    settings: CompilerSettings,
) -> impl Receiver<Result = CompilingReceiverResult<'a>, Node = Mig> + 'a {
    EGraph::<MigLanguage, _>::new(()).map(move |(mut graph, outputs)| {
        let t_runner = if settings.rewrite {
            let t_runner = std::time::Instant::now();
            let runner = Runner::default().with_egraph(graph).run(rules);
            let t_runner = t_runner.elapsed().as_millis();
            if settings.verbose {
                println!("== Runner Report");
                runner.print_report();
            }
            graph = runner.egraph;
            t_runner
        } else {
            0
        };

        let mut t_extractor = 0;
        let mut t_compiler = 0;

        let output = CompilerOutput::new(
            graph,
            |graph| {
                let start_time = Instant::now();
                let extractor = OptExtractor::new(graph, CompilingCostFunction { architecture });
                t_extractor = start_time.elapsed().as_millis();
                OptExtractionNetwork(extractor, outputs)
            },
            |ntk| {
                let start_time = Instant::now();
                let program = compile(architecture, &ntk.with_backward_edges())
                    .expect("network should be compilable");
                t_compiler = start_time.elapsed().as_millis();
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
struct CompilerSettings {
    print_program: bool,
    verbose: bool,
    rewrite: bool,
}

#[repr(C)]
struct CompilerStatistics {
    egraph_classes: u64,
    egraph_nodes: u64,
    egraph_size: u64,

    instruction_count: u64,
    runtime_estimate: u64,
    energy_consumption_estimate: u64,

    t_runner: u64,
    t_extractor: u64,
    t_compiler: u64,
}

#[no_mangle]
extern "C" fn prada_rewrite_ffi(
    settings: CompilerSettings,
    receiver: MigReceiverFFI<()>,
) -> MigReceiverFFI<CompilerStatistics> {
    let receiver =
        compiling_receiver(&ARCHITECTURE, REWRITE_RULES.as_slice(), settings).map(|res| {
            res.output.borrow_ntk().send(receiver);
            CompilerStatistics::from_result(res)
        });
    MigReceiverFFI::new(receiver)
}

#[no_mangle]
extern "C" fn prada_compile_ffi(settings: CompilerSettings) -> MigReceiverFFI<CompilerStatistics> {
    env_logger::init();
    let receiver = compiling_receiver(&ARCHITECTURE, REWRITE_RULES.as_slice(), settings)
        .map(CompilerStatistics::from_result);
    MigReceiverFFI::new(receiver)
}

impl CompilerStatistics {
    fn from_result(res: CompilingReceiverResult) -> Self {
        let graph = res.output.borrow_graph();
        CompilerStatistics {
            egraph_classes: graph.number_of_classes() as u64,
            egraph_nodes: graph.total_number_of_nodes() as u64,
            egraph_size: graph.total_size() as u64,
            instruction_count: res.output.borrow_program().instructions.len() as u64,
            runtime_estimate: res.output.borrow_program().runtime_estimate,
            energy_consumption_estimate: res.output.borrow_program().energy_consumption_estimate,
            t_runner: res.t_runner as u64,
            t_extractor: res.t_extractor as u64,
            t_compiler: res.t_compiler as u64,
        }
    }
}
