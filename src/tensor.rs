use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Deref, Index, Range},
    sync::Arc,
};

use bytemuck::{AnyBitPattern, NoUninit};
use wgpu::{BufferDescriptor, COPY_BUFFER_ALIGNMENT, util::DownloadBuffer};

use crate::{
    Device, ElementWiseOperation, MatMulOperation, PairWiseFunction, PairWiseOperation,
    ReduceFunction, ReduceOperation,
    compute_graph::{AnyComputeKey, ComputeGraph},
    layout::Layout,
    slice::{Slice, SliceOperation},
};

pub trait DataType: NoUninit + AnyBitPattern + Debug + Display {
    const WGSL_TYPE: DataTypeEnum;
}

impl DataType for f32 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F32;
}

impl DataType for half::f16 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F16;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataTypeEnum {
    F32,
    F16,
}

impl DataTypeEnum {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataTypeEnum::F32 => "f32",
            DataTypeEnum::F16 => "f16",
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            DataTypeEnum::F32 => size_of::<f32>(),
            DataTypeEnum::F16 => size_of::<half::f16>(),
        }
    }
}

impl Display for DataTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone)]
pub(crate) struct TensorInfo {
    layout: Layout,
    datatype: DataTypeEnum,
}

#[derive(Clone)]
pub(crate) struct LazyTensorData {
    device: Device,
    info: TensorInfo,
    graph: ComputeGraph,
    key: AnyComputeKey,
}

impl LazyTensorData {
    pub(crate) fn new(data: TensorData) -> Self {
        let graph = ComputeGraph::new();
        let device = data.device.clone();
        let info = data.info.clone();
        let key = graph.create_tensor(data);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn element_wise(&self, function: ElementWiseOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_element_wise(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn pair_wise(&self, function: PairWiseOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_pair_wise(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn mat_mul(&self, function: MatMulOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_mat_mul(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn reduce(&self, function: ReduceOperation) -> Self {
        let graph = self.graph.clone();
        let device = self.device.clone();
        let info = self.info.clone();
        let key = graph.create_reduce(function);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn slice(&self, op: SliceOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let graph = self.graph.clone();
        let key = self.graph.create_slice(op);

        Self {
            device,
            info,
            graph,
            key: key.into(),
        }
    }

    pub(crate) fn materialize(&self) -> TensorData {
        self.graph.resolve(self.key, &self.device)
    }
}

#[derive(Clone)]
pub(crate) struct TensorData {
    device: Device,
    buffer: Arc<wgpu::Buffer>,
    info: TensorInfo,
}

impl TensorData {
    pub(crate) fn new_from_buffer(
        device: &Device,
        buffer: wgpu::Buffer,
        size: &[usize],
        datatype: DataTypeEnum,
    ) -> Self {
        Self {
            device: device.clone(),
            buffer: Arc::new(buffer),
            info: TensorInfo {
                layout: Layout::contiguous(size),
                datatype,
            },
        }
    }

    fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: &[usize],
    ) -> Self {
        // MODIFIED from: https://github.com/gfx-rs/wgpu/blob/d8833d079833c62b4fd00325d0ba08ec0c8bc309/wgpu/src/util/device.rs#L38
        fn create_aligned_buffer(
            element_size: u64,
            shape: &[usize],
            device: &Device,
        ) -> (wgpu::Buffer, u64) {
            let size = element_size * shape.iter().copied().product::<usize>() as u64;

            let padded_size = padded_tensor_size(size);

            let wgt_descriptor = BufferDescriptor {
                label: None,
                size: padded_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            };

            let buffer = device.wgpu_device().create_buffer(&wgt_descriptor);
            (buffer, size)
        }
        let (buffer, unpadded_size) = create_aligned_buffer(size_of::<D>() as u64, &shape, device);

        buffer.slice(..).get_mapped_range_mut()[..unpadded_size as usize]
            .iter_mut()
            .zip(data.flat_map(bytemuck::bytes_of))
            .for_each(|(dst, src)| *dst = *src);
        buffer.unmap();

        Self::new_from_buffer(device, buffer, &shape, D::WGSL_TYPE)
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Self {
        let layout = self.info.layout.slice(ranges);
        Self {
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            info: TensorInfo {
                layout,
                datatype: self.info.datatype,
            },
        }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.info.layout
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.info.datatype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

pub struct Tensor<const R: usize, D> {
    data: LazyTensorData,
    datatype: PhantomData<D>,
}

impl<const R: usize, D: DataType> From<TensorData> for Tensor<R, D> {
    fn from(value: TensorData) -> Self {
        Self {
            data: LazyTensorData::new(value),
            datatype: PhantomData,
        }
    }
}

impl<const R: usize, D> Clone for Tensor<R, D> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            datatype: PhantomData,
        }
    }
}

pub trait IntoTensor<const R: usize, D> {
    fn into_tensor(self, device: &Device) -> Tensor<R, D>;
}

impl<'a, I, D: DataType> IntoTensor<1, D> for I
where
    I: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<1, D> {
        let iter = self.into_iter();
        let size = iter.len();
        Tensor::new_inner(device, iter, [size])
    }
}

impl<'a, I, I2, D: DataType> IntoTensor<2, D> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<2, D> {
        let mut iter = self.into_iter().map(IntoIterator::into_iter).peekable();
        let size = iter.len();
        let second_size = iter.peek().map(ExactSizeIterator::len).unwrap_or_default();
        let iter = iter.flat_map(|i| {
            let size = i.len();
            if size != second_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {second_size}, but another inner iterator size was {size}");
            }
            i
        });
        Tensor::new_inner(device, iter, [size, second_size])
    }
}

impl<'a, I, I2, I3, D: DataType> IntoTensor<3, D> for I
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn into_tensor(self, device: &Device) -> Tensor<3, D> {
        let mut iter = self
            .into_iter()
            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
            .peekable();
        let mut shape = [iter.len(), 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek() {
                let size = iter.len();
                shape[2] = size;
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<D: DataType, const R: usize> Tensor<R, D> {
    pub fn new(device: &Device, data: impl IntoTensor<R, D>) -> Self {
        data.into_tensor(device)
    }

    fn new_inner<'a, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: [usize; R],
    ) -> Self {
        Self {
            data: LazyTensorData::new(TensorData::new_inner(device, data, &shape)),
            datatype: PhantomData,
        }
    }

    async fn as_slice_from_tensor_data(
        tensor: &TensorData,
    ) -> Result<TensorSlice<R, D>, wgpu::BufferAsyncError> {
        let buffer = tensor.buffer();
        let (sender, receiver) = futures_channel::oneshot::channel();
        DownloadBuffer::read_buffer(
            tensor.device.wgpu_device(),
            tensor.device.wgpu_queue(),
            &buffer.slice(..),
            move |result| {
                _ = sender.send(result);
            },
        );
        let downloaded = receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        Ok(TensorSlice::new(downloaded, tensor.layout().clone()))
    }

    pub async fn as_slice(&self) -> Result<TensorSlice<R, D>, wgpu::BufferAsyncError> {
        let tensor = self.data.materialize();
        Self::as_slice_from_tensor_data(&tensor).await
    }

    pub(crate) fn element_wise(&self, function: ElementWiseOperation) -> Self {
        Self {
            data: self.data.element_wise(function),
            datatype: PhantomData,
        }
    }

    pub(crate) fn pair_wise(&self, other: &Self, function: PairWiseFunction) -> Self {
        self.data.graph.merge(&other.data.graph);
        let operation = PairWiseOperation::new(function, self.data.key, other.data.key);
        Self {
            data: self.data.pair_wise(operation),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_mat_mul(&self, other: &Self) -> Self {
        self.data.graph.merge(&other.data.graph);
        let operation = MatMulOperation::new(self.data.key, other.data.key);

        Self {
            data: self.data.mat_mul(operation),
            datatype: PhantomData,
        }
    }

    pub(crate) fn reduce<const OUT: usize>(
        &self,
        function: ReduceFunction,
        dim: usize,
    ) -> Tensor<OUT, D> {
        Tensor {
            data: self
                .data
                .reduce(ReduceOperation::new(self.data.key, function, dim)),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_slice(&self, op: Slice) -> Self {
        Self {
            data: self.data.slice(SliceOperation::new(self.data.key, op)),
            datatype: PhantomData,
        }
    }

    pub(crate) fn key(&self) -> AnyComputeKey {
        self.data.key
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_slice() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice.get([0, 1]), None);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice.get([1, 1]), None);
    assert_eq!(as_slice.get([2, 0]), None);
    assert_eq!(as_slice.get([2, 1]), None);
}

pub struct TensorSlice<const R: usize, D> {
    buffer: DownloadBuffer,
    layout: Layout,
    datatype: PhantomData<D>,
}

impl<D: DataType + Debug> Debug for TensorSlice<1, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| self.get([i]).unwrap())
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<2, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| self.get([i, j]).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: DataType + Debug> Debug for TensorSlice<3, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| {
                        (0..shape[2])
                            .map(|k| self.get([i, j, k]).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<const R: usize, D: DataType + PartialEq> PartialEq for TensorSlice<R, D> {
    fn eq(&self, other: &Self) -> bool {
        let self_shape = self.layout.shape();
        let other_shape = other.layout.shape();
        if self_shape != other_shape {
            return false;
        }
        let mut index = [0; R];
        loop {
            for i in 0..R {
                index[i] += 1;
                if index[i] >= self_shape[i] {
                    index[i] = 0;
                    if i == R - 1 {
                        return true;
                    }
                    index[i + 1] += 1;
                } else {
                    break;
                }
            }
            if self.get(index) != other.get(index) {
                return false;
            }
        }
    }
}

pub(crate) fn padded_tensor_size(size: u64) -> u64 {
    // Valid vulkan usage is
    // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
    // 2. buffer size must be greater than 0.
    // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    let padded_size = ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT);
    padded_size
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_compare() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
        [[7., 8.], [7., 8.]],
        [[9., 10.], [9., 10.]],
        [[11., 12.], [11., 12.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice, as_slice);

    let other_slice = tensor.slice([0..1, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);

    let other_slice = tensor.slice([1..3, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);
}

impl<D: DataType, const R: usize> TensorSlice<R, D> {
    fn new(buffer: DownloadBuffer, layout: Layout) -> Self {
        Self {
            buffer,
            layout,
            datatype: PhantomData,
        }
    }

    fn as_slice(&self) -> &[D] {
        bytemuck::cast_slice(&self.buffer.deref()[self.layout.offset() * size_of::<D>()..])
    }
}

impl<D: DataType, const R: usize> TensorSlice<R, D> {
    fn get(&self, index: [usize; R]) -> Option<&D> {
        let mut index_sum = 0;
        let layout = &self.layout;
        for ((index_component, &stride), &size) in
            index.into_iter().zip(layout.strides()).zip(layout.shape())
        {
            if index_component >= size {
                return None;
            }
            index_sum += stride * index_component;
        }

        self.as_slice().get(index_sum)
    }
}

impl<D: DataType, const R: usize> Index<[usize; R]> for TensorSlice<R, D> {
    type Output = D;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let as_slice = tensor.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}
