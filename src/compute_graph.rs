use std::{
    collections::HashMap,
    sync::{Arc, RwLock, atomic::AtomicUsize},
};

use arc_swap::ArcSwap;
use wgpu::wgc::identity;

use crate::{
    ElementWiseOperation, MatMulOperation, PairWiseOperation, ReduceOperation,
    UntypedElementWiseKernel, UntypedPairWiseKernel, UntypedReduceKernel, matmul::UntypedMatMul,
    tensor::TensorData,
};
use tabbycat::{AttrList, Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList, SubGraph};
use tabbycat::{Graph, attributes::*};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ElementWiseComputeNodeKey(usize);
impl ElementWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PairWiseComputeNodeKey(usize);
impl PairWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MatMulComputeNodeKey(usize);
impl MatMulComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ReduceComputeNodeKey(usize);
impl ReduceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TensorComputeNodeKey(usize);
impl TensorComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AnyComputeKey {
    ElementWiseComputeNodeKey(ElementWiseComputeNodeKey),
    PairWiseComputeNodeKey(PairWiseComputeNodeKey),
    MatMulComputeNodeKey(MatMulComputeNodeKey),
    ReduceComputeNodeKey(ReduceComputeNodeKey),
    TensorComputeNodeKey(TensorComputeNodeKey),
}

impl From<ElementWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: ElementWiseComputeNodeKey) -> Self {
        Self::ElementWiseComputeNodeKey(value)
    }
}

impl From<PairWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: PairWiseComputeNodeKey) -> Self {
        Self::PairWiseComputeNodeKey(value)
    }
}

impl From<MatMulComputeNodeKey> for AnyComputeKey {
    fn from(value: MatMulComputeNodeKey) -> Self {
        Self::MatMulComputeNodeKey(value)
    }
}

impl From<ReduceComputeNodeKey> for AnyComputeKey {
    fn from(value: ReduceComputeNodeKey) -> Self {
        Self::ReduceComputeNodeKey(value)
    }
}

impl From<TensorComputeNodeKey> for AnyComputeKey {
    fn from(value: TensorComputeNodeKey) -> Self {
        Self::TensorComputeNodeKey(value)
    }
}

#[derive(Clone, Default)]
pub(crate) struct ComputeGraph {
    inner: Arc<ArcSwap<RwLock<ComputeGraphInner>>>,
}

impl ComputeGraph {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let write = self.inner.load();
        let mut inner = write.write().unwrap();
        f(&mut *inner)
    }

    pub(crate) fn merge(&self, other: &Self) {
        self.with_mut(|inner| {
            other.with_mut(|other_inner| {
                inner.element_wise.extend(other_inner.element_wise.drain());
                inner.pair_wise.extend(other_inner.pair_wise.drain());
                inner.mat_mul.extend(other_inner.mat_mul.drain());
                inner.reduce.extend(other_inner.reduce.drain());
                inner.tensor.extend(other_inner.tensor.drain());
            })
        });
        other.inner.store(self.inner.load_full());
    }

    pub(crate) fn create_element_wise(
        &self,
        function: ElementWiseOperation,
    ) -> ElementWiseComputeNodeKey {
        let id = ElementWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.element_wise.insert(id, function));
        id
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> PairWiseComputeNodeKey {
        let id = PairWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.pair_wise.insert(id, function));
        id
    }

    pub(crate) fn create_mat_mul(&self, function: MatMulOperation) -> MatMulComputeNodeKey {
        let id = MatMulComputeNodeKey::new();
        self.with_mut(|inner| inner.mat_mul.insert(id, function));
        id
    }

    pub(crate) fn create_reduce(&self, function: ReduceOperation) -> ReduceComputeNodeKey {
        let id = ReduceComputeNodeKey::new();
        self.with_mut(|inner| inner.reduce.insert(id, function));
        id
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> TensorComputeNodeKey {
        let id = TensorComputeNodeKey::new();
        self.with_mut(|inner| inner.tensor.insert(id, info));
        id
    }
}

#[derive(Default)]
struct ComputeGraphInner {
    element_wise: HashMap<ElementWiseComputeNodeKey, ElementWiseOperation>,
    pair_wise: HashMap<PairWiseComputeNodeKey, PairWiseOperation>,
    mat_mul: HashMap<MatMulComputeNodeKey, MatMulOperation>,
    reduce: HashMap<ReduceComputeNodeKey, ReduceOperation>,
    tensor: HashMap<TensorComputeNodeKey, TensorData>,
}

impl ComputeGraphInner {
    pub(crate) fn resolve(&self, key: AnyComputeKey) -> TensorData {
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.resolve_element_wise(element_wise_compute_node_key)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.resolve_pair_wise(pair_wise_compute_node_key)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.resolve_mat_mul(mat_mul_compute_node_key)
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.resolve_reduce(reduce_compute_node_key)
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.resolve_tensor(tensor_compute_node_key)
            }
        }
    }

    fn resolve_element_wise(&self, key: ElementWiseComputeNodeKey) -> TensorData {
        let operation = self.element_wise.get(&key).unwrap();

        let input = self.resolve(operation.value);
        let kernel =
            UntypedElementWiseKernel::new(vec![operation.function.clone()], input.datatype());
        kernel.run_with_query(&input, None);
        input
    }

    fn resolve_pair_wise(&self, key: PairWiseComputeNodeKey) -> TensorData {
        let operation = self.pair_wise.get(&key).unwrap();

        let first = self.resolve(operation.first);
        let second = self.resolve(operation.second);
        let kernel = UntypedPairWiseKernel::new(operation.function.clone(), first.datatype());
        kernel.run_with_query(&first, &second, None);
        first
    }

    fn resolve_mat_mul(&self, key: MatMulComputeNodeKey) -> TensorData {
        let operation = self.mat_mul.get(&key).unwrap();

        let first = self.resolve(operation.first);
        let second = self.resolve(operation.second);
        let kernel = UntypedMatMul::new(first.datatype());
        kernel.run_with_query(&first, &second, None);
        first
    }

    fn resolve_reduce(&self, key: ReduceComputeNodeKey) -> TensorData {
        let operation = self.reduce.get(&key).unwrap();

        let input = self.resolve(operation.value);
        let kernel = UntypedReduceKernel::new(operation.function.clone(), input.datatype());
        kernel.run_with_query(&input, 0, None);
        input
    }

    fn resolve_tensor(&self, key: TensorComputeNodeKey) -> TensorData {
        self.tensor.get(&key).unwrap().clone()
    }

    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut statements = Vec::new();
        self.add_node_to_graph(&mut statements, root);
        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .stmts(StmtList::new().extend(statements))
            .build()
            .unwrap()
    }

    fn add_node_to_graph(&self, graph: &mut Vec<Stmt>, key: AnyComputeKey) -> Identity {
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.add_element_wise_to_graph(graph, element_wise_compute_node_key)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.add_pair_wise_to_graph(graph, pair_wise_compute_node_key)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.add_mat_mul_to_graph(graph, mat_mul_compute_node_key)
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.add_reduce_to_graph(graph, reduce_compute_node_key)
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.add_tensor_to_graph(graph, tensor_compute_node_key)
            }
        }
    }

    fn add_element_wise_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ElementWiseComputeNodeKey,
    ) -> Identity {
        let operation = self.element_wise.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value);
        let id = Identity::id(operation.function.name()).unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(input, None),
        ));
        id
    }

    fn add_pair_wise_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: PairWiseComputeNodeKey,
    ) -> Identity {
        let operation = self.pair_wise.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first);
        let second = self.add_node_to_graph(graph, operation.second);
        let id = Identity::id(operation.function.name()).unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(first, None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(second, None),
        ));
        id
    }

    fn add_mat_mul_to_graph(&self, graph: &mut Vec<Stmt>, key: MatMulComputeNodeKey) -> Identity {
        let operation = self.mat_mul.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first);
        let second = self.add_node_to_graph(graph, operation.second);
        let id = Identity::id("matmul").unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(first, None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(second, None),
        ));
        id
    }

    fn add_reduce_to_graph(&self, graph: &mut Vec<Stmt>, key: ReduceComputeNodeKey) -> Identity {
        let operation = self.reduce.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value);
        let id = Identity::id(operation.function.name()).unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(input, None),
        ));
        id
    }

    fn add_tensor_to_graph(&self, graph: &mut Vec<Stmt>, key: TensorComputeNodeKey) -> Identity {
        let id = Identity::id(format!("tensor_{}", key.0)).unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(id.clone(), None),
        ));
        id
    }
}
