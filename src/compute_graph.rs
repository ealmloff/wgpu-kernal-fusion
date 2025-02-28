use std::{
    collections::HashMap,
    sync::{Arc, RwLock, atomic::AtomicUsize},
};

use arc_swap::ArcSwap;
use wgpu::CommandEncoder;

use crate::{
    Device, ElementWiseFunction, ElementWiseOperation, MatMulOperation, PairWiseOperation,
    ReduceOperation, UntypedElementWiseKernel, UntypedPairWiseKernel, UntypedReduceKernel,
    element_wise, matmul::UntypedMatMul, slice::SliceOperation, tensor::TensorData,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ElementWiseComputeNodeKey(usize);
impl ElementWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PairWiseComputeNodeKey(usize);
impl PairWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatMulComputeNodeKey(usize);
impl MatMulComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceComputeNodeKey(usize);
impl ReduceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SliceComputeNodeKey(usize);
impl SliceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct TensorComputeNodeKey(usize);
impl TensorComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AnyComputeKey {
    ElementWiseComputeNodeKey(ElementWiseComputeNodeKey),
    PairWiseComputeNodeKey(PairWiseComputeNodeKey),
    MatMulComputeNodeKey(MatMulComputeNodeKey),
    ReduceComputeNodeKey(ReduceComputeNodeKey),
    SliceComputeNodeKey(SliceComputeNodeKey),
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

impl From<SliceComputeNodeKey> for AnyComputeKey {
    fn from(value: SliceComputeNodeKey) -> Self {
        Self::SliceComputeNodeKey(value)
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
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        self.with_mut(|inner| {
            other.with_mut(|other_inner| {
                inner.element_wise.extend(other_inner.element_wise.drain());
                inner.pair_wise.extend(other_inner.pair_wise.drain());
                inner.mat_mul.extend(other_inner.mat_mul.drain());
                inner.reduce.extend(other_inner.reduce.drain());
                inner.slice.extend(other_inner.slice.drain());
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

    pub(crate) fn create_slice(&self, op: SliceOperation) -> SliceComputeNodeKey {
        let id = SliceComputeNodeKey::new();
        self.with_mut(|inner| inner.slice.insert(id, op));
        id
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> TensorComputeNodeKey {
        let id = TensorComputeNodeKey::new();
        self.with_mut(|inner| inner.tensor.insert(id, info));
        id
    }

    pub(crate) fn resolve(&self, key: AnyComputeKey, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&Default::default());
        let data = self.with_mut(|inner| inner.resolve(key, &mut encoder));
        device.wgpu_queue().submit(Some(encoder.finish()));
        data
    }
}

#[derive(Default)]
struct ComputeGraphInner {
    element_wise: HashMap<ElementWiseComputeNodeKey, ElementWiseOperation>,
    pair_wise: HashMap<PairWiseComputeNodeKey, PairWiseOperation>,
    mat_mul: HashMap<MatMulComputeNodeKey, MatMulOperation>,
    reduce: HashMap<ReduceComputeNodeKey, ReduceOperation>,
    slice: HashMap<SliceComputeNodeKey, SliceOperation>,
    tensor: HashMap<TensorComputeNodeKey, TensorData>,
}

impl ComputeGraphInner {
    pub(crate) fn resolve(
        &self,
        key: AnyComputeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let graph = self.graphvis(key);
        println!("{graph}");
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.resolve_element_wise(element_wise_compute_node_key, command_encoder)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.resolve_pair_wise(pair_wise_compute_node_key, command_encoder)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.resolve_mat_mul(mat_mul_compute_node_key, command_encoder)
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.resolve_reduce(reduce_compute_node_key, command_encoder)
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.resolve_tensor(tensor_compute_node_key, command_encoder)
            }
            AnyComputeKey::SliceComputeNodeKey(slice_compute_node_key) => {
                self.resolve_slice(slice_compute_node_key, command_encoder)
            }
        }
    }

    fn collect_element_wise_ops(
        &self,
        key: ElementWiseComputeNodeKey,
    ) -> (Vec<ElementWiseFunction>, AnyComputeKey) {
        let mut functions = Vec::new();
        let mut current_key = AnyComputeKey::ElementWiseComputeNodeKey(key);
        while let AnyComputeKey::ElementWiseComputeNodeKey(key) = current_key {
            let operation = self.element_wise.get(&key).unwrap();
            functions.push(operation.function.clone());
            current_key = operation.value;
        }
        (functions, current_key)
    }

    fn resolve_element_wise(
        &self,
        key: ElementWiseComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        // First collect all element wise ops in this chain
        let (functions, input) = self.collect_element_wise_ops(key);

        // Merge into the output of the reduce kernel if possible
        if let AnyComputeKey::ReduceComputeNodeKey(key) = input {
            self.resolve_reduce_then(key, functions, command_encoder)
        }
        // Merge into the output of the pair wise kernel if possible
        else if let AnyComputeKey::PairWiseComputeNodeKey(key) = input {
            self.resolve_pair_wise_then(key, functions, command_encoder)
        } else {
            let input = self.resolve(input, &mut *command_encoder);
            let kernel = UntypedElementWiseKernel::new(functions, input.datatype());
            kernel
                .run_with_query(&input, None, command_encoder)
                .unwrap_or(input)
        }
    }

    fn resolve_pair_wise(
        &self,
        key: PairWiseComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        self.resolve_pair_wise_then(key, Vec::new(), command_encoder)
    }

    fn resolve_pair_wise_then(
        &self,
        key: PairWiseComputeNodeKey,
        then: Vec<ElementWiseFunction>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.pair_wise.get(&key).unwrap();

        let mut first_input = operation.first;
        let first_pre_element_wise =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = operation.first {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                first_input = element_wise_input;
                functions
            } else {
                Vec::new()
            };
        let mut second_input = operation.second;
        let second_pre_element_wise =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = operation.second {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                second_input = element_wise_input;
                functions
            } else {
                Vec::new()
            };

        let first = self.resolve(first_input, &mut *command_encoder);
        let second = self.resolve(second_input, &mut *command_encoder);
        let mut kernel = UntypedPairWiseKernel::new(operation.function.clone(), first.datatype());
        kernel.set_pre_element_wise([
            UntypedElementWiseKernel::new(first_pre_element_wise, first.datatype()),
            UntypedElementWiseKernel::new(second_pre_element_wise, first.datatype()),
        ]);
        kernel.set_post_element_wise(UntypedElementWiseKernel::new(then, first.datatype()));
        kernel.run_with_query(&first, &second, None, command_encoder);
        second
    }

    fn resolve_mat_mul(
        &self,
        key: MatMulComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.mat_mul.get(&key).unwrap();

        let first = self.resolve(operation.first, &mut *command_encoder);
        let second = self.resolve(operation.second, &mut *command_encoder);
        let kernel = UntypedMatMul::new(first.datatype());
        kernel.run_with_query(&first, &second, None, command_encoder)
    }

    fn resolve_reduce(
        &self,
        key: ReduceComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        self.resolve_reduce_then(key, Vec::new(), command_encoder)
    }

    fn resolve_reduce_then(
        &self,
        key: ReduceComputeNodeKey,
        then: Vec<ElementWiseFunction>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.reduce.get(&key).unwrap();
        let mut input = operation.value;

        let element_wise_before =
            if let AnyComputeKey::ElementWiseComputeNodeKey(key) = operation.value {
                let (functions, element_wise_input) = self.collect_element_wise_ops(key);
                input = element_wise_input;
                functions
            } else {
                Vec::new()
            };

        let input = self.resolve(input, &mut *command_encoder);
        let mut kernel = UntypedReduceKernel::new(operation.function.clone(), input.datatype());
        let element_wise_before =
            element_wise::UntypedElementWiseKernel::new(element_wise_before, input.datatype());
        let element_wise_after =
            element_wise::UntypedElementWiseKernel::new(then, input.datatype());
        kernel.set_post_element_wise(element_wise_after);
        kernel.set_pre_element_wise(element_wise_before);
        kernel.run_with_query(&input, operation.axis, None, command_encoder)
    }

    fn resolve_slice(
        &self,
        key: SliceComputeNodeKey,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let operation = self.slice.get(&key).unwrap();
        let input = self.resolve(operation.input, &mut *command_encoder);

        operation.run(&input)
    }

    fn resolve_tensor(&self, key: TensorComputeNodeKey, _: &mut CommandEncoder) -> TensorData {
        self.tensor.get(&key).unwrap().clone()
    }

    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut statements = Vec::new();
        self.add_node_to_graph(&mut statements, root);
        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::id("ComputeGraph").unwrap())
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
            AnyComputeKey::SliceComputeNodeKey(slice_compute_node_key) => {
                self.add_slice_to_graph(graph, slice_compute_node_key)
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
            Edge::head_node(input, None).arrow_to_node(id.clone(), None),
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
            Edge::head_node(first, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(second, None).arrow_to_node(id.clone(), None),
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
            Edge::head_node(first, None).arrow_to_node(id.clone(), None),
        ));
        graph.push(Stmt::Edge(
            Edge::head_node(second, None).arrow_to_node(id.clone(), None),
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

    fn add_slice_to_graph(&self, graph: &mut Vec<Stmt>, key: SliceComputeNodeKey) -> Identity {
        let operation = self.slice.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input);
        let id = Identity::id("slice").unwrap();
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        graph.push(Stmt::Edge(
            Edge::head_node(id.clone(), None).arrow_to_node(input.into(), None),
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
        id
    }
}
