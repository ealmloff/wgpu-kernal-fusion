use std::ops::Range;

use crate::{
    DataType, Layout, Tensor, TensorData, compute_graph::AnyComputeKey, slice_shape, slice_strides,
};

pub(crate) struct MapLayoutOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) map_size: Box<dyn Fn(&[usize]) -> Box<[usize]>>,
    pub(crate) map_stride: Box<dyn Fn(usize, &[usize]) -> (usize, Box<[usize]>)>,
}

impl MapLayoutOperation {
    pub fn new(
        input: AnyComputeKey,
        map_size: impl Fn(&[usize]) -> Box<[usize]> + 'static,
        map_stride: impl Fn(usize, &[usize]) -> (usize, Box<[usize]>) + 'static,
    ) -> Self {
        Self {
            input,
            map_size: Box::new(map_size),
            map_stride: Box::new(map_stride),
        }
    }

    pub fn run(&self, tensor: &TensorData) -> TensorData {
        TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            self.map_layout(tensor.layout()),
            tensor.datatype(),
        )
    }

    pub fn map_layout(&self, layout: &Layout) -> Layout {
        let (offset, strides) = (self.map_stride)(layout.offset(), layout.strides());
        Layout::from_parts(offset, (self.map_size)(layout.shape()), strides)
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn slice(&self, slices: [Range<usize>; R]) -> Tensor<R, T> {
        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            {
                let slices = slices.clone();
                move |shape| slice_shape(&slices, shape)
            },
            move |offset, strides| slice_strides(&slices, offset, strides),
        ))
    }

    pub fn transpose(&self, first_axis: usize, second_axis: usize) -> Tensor<R, T> {
        self.add_map_layout(MapLayoutOperation::new(
            self.key(),
            move |shape| {
                let mut shape: Box<[usize]> = shape.into();
                shape.swap(first_axis, second_axis);
                shape
            },
            move |offset, strides| {
                let mut strides: Box<[usize]> = strides.into();
                strides.swap(first_axis, second_axis);
                (offset, strides)
            },
        ))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_transpose() {
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
    let transposed = tensor.transpose(0, 1);
    let as_slice = transposed.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 3.);
    assert_eq!(as_slice[[0, 2]], 5.);
    assert_eq!(as_slice[[1, 0]], 2.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[1, 2]], 6.);
}
