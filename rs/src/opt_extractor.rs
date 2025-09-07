use std::{fmt::Debug, ops::Index};

use eggmock::{
    egg::{Analysis, EClass, EGraph, Id, Language},
    EggIdToSignal, Network, NetworkLanguage,
};
use rustc_hash::FxHashMap;

pub trait OptCostFunction<L: Language, A: Analysis<L>> {
    type Cost: PartialOrd + Debug + Clone;

    fn cost<C>(&mut self, eclass: &EClass<L, A::Data>, enode: &L, costs: C) -> Option<Self::Cost>
    where
        C: FnMut(Id) -> Self::Cost;
}

/// An extractor heavily inspired by egg's [Extractor](eggmock::egg::Extractor), which allows
/// ignoring certain nodes by returning [None] from their cost function.
pub struct OptExtractor<'g, CF: OptCostFunction<L, A>, L: Language, A: Analysis<L>> {
    graph: &'g EGraph<L, A>,
    cost_fn: CF,
    costs: FxHashMap<Id, (CF::Cost, L)>,
}

impl<'g, CF: OptCostFunction<L, A>, L: Language, A: Analysis<L>> OptExtractor<'g, CF, L, A> {
    pub fn new(graph: &'g EGraph<L, A>, cost_fn: CF) -> Self {
        let mut extractor = Self {
            graph,
            cost_fn,
            costs: FxHashMap::default(),
        };
        extractor.find_costs();
        extractor
    }

    pub fn find_best_node(&self, class: Id) -> Option<&L> {
        self.costs
            .get(&self.graph.find(class))
            .map(|(_, node)| node)
    }

    fn find_costs(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            for class in self.graph.classes() {
                let new_cost = self.determine_class_costs(class);
                match (self.costs.get(&class.id), new_cost) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        changed = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        changed = true;
                    }
                    _ => (),
                }
            }
        }
    }

    fn determine_class_costs(&mut self, class: &EClass<L, A::Data>) -> Option<(CF::Cost, L)> {
        class
            .iter()
            .map(|node| (self.opt_node_cost(node, class), node))
            .filter_map(|(cost, node)| cost.map(|cost| (cost, node)))
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .map(|(cost, node)| (cost, node.clone()))
    }

    fn opt_node_cost(&mut self, node: &L, class: &EClass<L, A::Data>) -> Option<CF::Cost> {
        if node.all(|id| self.costs.contains_key(&id)) {
            self.cost_fn
                .cost(class, node, |id| self.costs[&self.graph.find(id)].0.clone())
        } else {
            None
        }
    }
}

pub struct OptExtractionNetwork<E>(pub E, pub Vec<Id>);

impl<CF, L, A> Network for OptExtractionNetwork<OptExtractor<'_, CF, L, A>>
where
    CF: OptCostFunction<L, A>,
    L: NetworkLanguage,
    A: Analysis<L>,
{
    type Node = L::Node;

    fn outputs(&self) -> impl Iterator<Item = eggmock::Signal> {
        self.1.iter().map(|id| IndexWrapper(&self.0).to_signal(*id))
    }

    fn node(&self, id: eggmock::Id) -> Self::Node {
        self.0
            .find_best_node(id.into())
            .expect("class should be extractable")
            .to_node(|id| IndexWrapper(&self.0).to_signal(id))
            .expect("id should point to a non-not node")
    }
}

struct IndexWrapper<'e, E>(&'e E);

impl<CF, L, A> Index<Id> for IndexWrapper<'_, OptExtractor<'_, CF, L, A>>
where
    CF: OptCostFunction<L, A>,
    L: NetworkLanguage,
    A: Analysis<L>,
{
    type Output = L;

    fn index(&self, index: Id) -> &Self::Output {
        self.0.find_best_node(index).expect("class not extractable")
    }
}
