use std::sync::OnceLock;

use wgpu::CommandEncoder;

use crate::{
    PerformanceQueries, TILE_SIZE, Tensor, TensorData, compute_graph::AnyComputeKey,
    visit_tiled::VisitTiledKernel,
};

pub(crate) struct ResizeOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) new_shape: Vec<usize>,
}

impl ResizeOperation {
    pub fn new(input: AnyComputeKey, new_shape: Vec<usize>) -> Self {
        Self { input, new_shape }
    }
}

pub(crate) struct UntypedResizeKernel {
    new_shape: Box<[usize]>,
    sparse_kernel: OnceLock<VisitTiledKernel>,
}

impl UntypedResizeKernel {
    pub(crate) fn new(new_shape: &[usize]) -> Self {
        Self {
            new_shape: new_shape.into(),
            sparse_kernel: OnceLock::new(),
        }
    }

    pub fn run_with_query(
        &self,
        input: &TensorData,
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) -> TensorData {
        let rank = input.layout().rank();
        let datatype = input.datatype();

        let create_kernel = || {
            let datatypes = vec![datatype; 2];

            VisitTiledKernel::new(
                rank as u32,
                TILE_SIZE,
                false,
                datatypes,
                |_, indexes, tensors| {
                    let in_index = &indexes[0];
                    let out_index = &indexes[1];
                    let in_tensor = &tensors[0];
                    let out_tensor = &tensors[1];
                    format!("{out_tensor}[{out_index}] = {in_tensor}[{in_index}];")
                },
            )
        };
        let kernel = self.sparse_kernel.get_or_init(create_kernel);
        let output = TensorData::new_for_shape(input.device(), &self.new_shape, datatype);
        let tensors = vec![input, &output];
        kernel.run_with_query(tensors, query, command_encoder);
        output
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn resize(&self, new_shape: [usize; R]) -> Tensor<R, T> {
        let new_shape = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(input, new_shape))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_resize() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.resize([30, 20]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}
