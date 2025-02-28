use super::visit::VisitComputeGraph;
use super::{
    AnyComputeKey, ComputeGraphInner, ElementWiseComputeNodeKey, MatMulComputeNodeKey,
    PairWiseComputeNodeKey, ReduceComputeNodeKey, SliceComputeNodeKey, TensorComputeNodeKey,
    layout_pass,
};
use tabbycat::Graph;
use tabbycat::{Edge, GraphBuilder, GraphType, Identity, Stmt, StmtList};

impl ComputeGraphInner {
    pub(crate) fn graphvis(&self, root: AnyComputeKey) -> Graph {
        let mut layout_pass = layout_pass::LayoutPass::default();
        layout_pass.visit(self, root);
        let mut statements = Vec::new();
        self.add_node_to_graph(&mut statements, root, &layout_pass);
        GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::quoted("ComputeGraph"))
            .stmts(StmtList::new().extend(statements))
            .build()
            .unwrap()
    }

    fn add_node_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: AnyComputeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        match key {
            AnyComputeKey::ElementWiseComputeNodeKey(element_wise_compute_node_key) => {
                self.add_element_wise_to_graph(graph, element_wise_compute_node_key, layout_pass)
            }
            AnyComputeKey::PairWiseComputeNodeKey(pair_wise_compute_node_key) => {
                self.add_pair_wise_to_graph(graph, pair_wise_compute_node_key, layout_pass)
            }
            AnyComputeKey::MatMulComputeNodeKey(mat_mul_compute_node_key) => {
                self.add_mat_mul_to_graph(graph, mat_mul_compute_node_key, layout_pass)
            }
            AnyComputeKey::ReduceComputeNodeKey(reduce_compute_node_key) => {
                self.add_reduce_to_graph(graph, reduce_compute_node_key, layout_pass)
            }
            AnyComputeKey::TensorComputeNodeKey(tensor_compute_node_key) => {
                self.add_tensor_to_graph(graph, tensor_compute_node_key, layout_pass)
            }
            AnyComputeKey::SliceComputeNodeKey(slice_compute_node_key) => {
                self.add_slice_to_graph(graph, slice_compute_node_key, layout_pass)
            }
        }
    }

    fn add_element_wise_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ElementWiseComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let operation = self.element_wise.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value, layout_pass);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("{} ({})", operation.function.name(), output_layout));
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
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let operation = self.pair_wise.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first, layout_pass);
        let second = self.add_node_to_graph(graph, operation.second, layout_pass);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("{} ({})", operation.function.name(), output_layout));
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

    fn add_mat_mul_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: MatMulComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let operation = self.mat_mul.get(&key).unwrap();
        let first = self.add_node_to_graph(graph, operation.first, layout_pass);
        let second = self.add_node_to_graph(graph, operation.second, layout_pass);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("matmul ({})", output_layout));
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

    fn add_reduce_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: ReduceComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let operation = self.reduce.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.value, layout_pass);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("{} ({})", operation.function.name(), output_layout));
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

    fn add_slice_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: SliceComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let operation = self.slice.get(&key).unwrap();
        let input = self.add_node_to_graph(graph, operation.input, layout_pass);
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("slice ({})", output_layout));
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

    fn add_tensor_to_graph(
        &self,
        graph: &mut Vec<Stmt>,
        key: TensorComputeNodeKey,
        layout_pass: &layout_pass::LayoutPass,
    ) -> Identity {
        let output_layout = layout_pass.output_layout.get(&key.into()).unwrap();
        let id = Identity::quoted(format!("tensor_{} ({})", key.0, output_layout));
        graph.push(Stmt::Node {
            id: id.clone(),
            port: None,
            attr: None,
        });
        id
    }
}
