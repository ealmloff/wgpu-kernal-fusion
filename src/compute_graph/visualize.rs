use super::{
    AnyComputeKey, ComputeGraphInner, ElementWiseComputeNodeKey, MatMulComputeNodeKey,
    PairWiseComputeNodeKey, ReduceComputeNodeKey, SliceComputeNodeKey, TensorComputeNodeKey,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

impl ComputeGraphInner {
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
