use crate::opt_extractor::OptCostFunction;
use crate::prada::architecture::PRADAArchitecture;
use eggmock::egg::{Analysis, EClass, Id, Language};
use eggmock::MigLanguage;
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops;
use std::rc::Rc;

pub struct CompilingCostFunction<'a> {
    pub architecture: &'a PRADAArchitecture,
}

#[derive(Debug,Copy,Clone)]
pub struct CompilingCost {
    /// in ns (lower is better)
    /// estimated by summing up all latencies
    pub runtime: u64,
    /// in mJ/KOps (lower is better)
    pub energy_consumption: u64,
}

impl<A: Analysis<MigLanguage>> OptCostFunction<MigLanguage, A> for CompilingCostFunction<'_> {
    type Cost = Rc<CompilingCost>;

    fn cost<C>(
        &mut self,
        eclass: &EClass<MigLanguage, A::Data>,
        enode: &MigLanguage,
        mut costs: C,
    ) -> Option<Self::Cost>
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // detect self-cycles, other cycles will be detected by compiling, which will result in an
        // error
        if enode.children().contains(&eclass.id) {
            return None;
        }
        let root = enode.clone();
        let op_cost = match enode {
            MigLanguage::False | MigLanguage::Input(_) => CompilingCost::leaf(root),
            MigLanguage::Not(_) => {
                CompilingCost {
                    runtime: 35,
                    energy_consumption: 100,
                }
            }
            MigLanguage::Maj(_) => CompilingCost {
                runtime: 49,
                energy_consumption: 150,
            },
        };
        Some(Rc::new(enode.fold(op_cost, |sum, id| sum + *(costs(id)))))
    }
}

impl CompilingCost {
    pub fn leaf(root: MigLanguage) -> Self {
        Self {
            runtime: 0,
            energy_consumption: 0
        }
    }
}

/// Needed to implement `enode.fold()` for computing overall cost from node together with its children
impl ops::Add<CompilingCost> for CompilingCost {
    type Output = CompilingCost;

    fn add(self, rhs: CompilingCost) -> Self::Output {
        CompilingCost {
            // both values are monotonically increasing
            runtime: self.runtime + rhs.runtime, // monotonically decreasing
            energy_consumption: self.energy_consumption + rhs.energy_consumption,
        }
    }
}

impl Sum for CompilingCost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(CompilingCost{runtime: 0, energy_consumption: 0 }, |acc, x| CompilingCost{ runtime: acc.runtime + x.runtime, energy_consumption: acc.energy_consumption + x.energy_consumption })
    }
}

impl PartialEq for CompilingCost {
    fn eq(&self, other: &Self) -> bool {
        self.runtime.eq(&other.runtime) & self.energy_consumption.eq(&other.energy_consumption)
    }
}

impl PartialOrd for CompilingCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.runtime.eq(&other.runtime) {
            self.energy_consumption.partial_cmp(&other.energy_consumption)
        } else {
            self.runtime.partial_cmp(&other.runtime)
        }
    }
}
