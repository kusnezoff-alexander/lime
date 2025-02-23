use rustc_hash::FxHashSet;
use super::{Program, Row};

pub fn optimize(program: &mut Program) {
    let mut opt = Optimization {
        liveness: row_liveness(program),
        program
    };
    opt.remove_unused_results();
}

pub struct Optimization<'p, 'a> {
    liveness: Vec<FxHashSet<Row>>,
    program: &'p mut Program<'a>,
}

impl Optimization<'_, '_> {
    fn remove_unused_results(&mut self) {
        let mut i = 0;
        let mut liveness_i = 0;
        'outer: while liveness_i < self.liveness.len()-1 {
            let instruction = self.program.instructions[i];
            let liveness_after = &self.liveness[liveness_i+1];

            // if no overridden row is live after this instruction, we can remove the instruction
            for row in instruction.overridden_rows(self.program.architecture) {
                if liveness_after.contains(&row) || matches!(row, Row::Out(_)) {
                    i += 1;
                    liveness_i += 1;
                    continue 'outer;
                }
            }

            liveness_i += 1;
            self.program.instructions.remove(i);
        }
        self.liveness = row_liveness(self.program);
    }
}

/// Returns a vector of the same length as the program where the entry at the index of an
/// instruction the set at that index contains the rows that are live just before the instruction
fn row_liveness(program: &Program) -> Vec<FxHashSet<Row>> {
    let mut result = Vec::with_capacity(program.instructions.len());

    let mut currently_live = FxHashSet::default();
    for instruction in program.instructions.iter().rev() {
        for row in instruction.overridden_rows(program.architecture) {
            currently_live.remove(&row);
        }
        currently_live.extend(instruction.input_operands(program.architecture).map(|addr| addr.row()));
        result.push(currently_live.clone());
    }

    result.reverse();
    result
}
