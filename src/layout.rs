use std::{marker::PhantomData, ops::Range};

use wgpu::naga::common::wgsl;

use crate::{tensor::DataType, Tensor};

pub(crate) const TILE_SIZE: u32 = 8;

fn continuous_strides<const R: usize>(shape: [usize; R]) -> [usize; R] {
    let mut acc = 1;
    let mut strides = [0; R];
    for i in (0..R).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

#[derive(Clone)]
pub(crate) struct TensorLayout<const R: usize> {
    pub(crate) data: Box<[u32]>,
}

impl<const R: usize> TensorLayout<R> {
    pub(crate) fn for_tensor<D: DataType>(tensor: &Tensor<R, D>) -> Self {
        let layout = *tensor.layout();
        layout.into()
    }

    pub(crate) fn wgsl_type_definition(kernel: &mut String) {
        kernel.push_str("struct TensorLayout {\n");
        for i in 0..R {
            kernel.push_str(&format!("\tstride_{}: u32,\n", i));
        }
        for i in 0..R {
            kernel.push_str(&format!("\tshape_{}: u32,\n", i));
        }
        kernel.push_str(&format!("\toffset: u32,\n"));
        kernel.push_str("}\n");
    }
}

impl<const R: usize, D> From<Layout<R, D>> for TensorLayout<R> {
    fn from(layout: Layout<R, D>) -> Self {
        let data = layout
            .strides
            .iter()
            .map(|x| *x as u32)
            .chain(layout.shape.iter().map(|x| *x as u32))
            .chain(std::iter::once(layout.offset as u32))
            .collect();
        Self { data }
    }
}

pub(super) struct Layout<const R: usize, D> {
    offset: usize,
    shape: [usize; R],
    strides: [usize; R],
    data_type: PhantomData<D>,
}

impl<const R: usize, D> Clone for Layout<R, D> {
    fn clone(&self) -> Self {
        Self {
            offset: self.offset,
            shape: self.shape,
            strides: self.strides,
            data_type: PhantomData,
        }
    }
}
impl<const R: usize, D> Copy for Layout<R, D> {}

impl<const R: usize, D> Layout<R, D> {
    pub fn contiguous(shape: [usize; R]) -> Self {
        let strides = continuous_strides(shape);
        Self {
            offset: 0,
            shape,
            strides,
            data_type: PhantomData,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == continuous_strides(self.shape)
    }

    pub fn slice(&self, index: [Range<usize>; R]) -> Self {
        let shape = std::array::from_fn(|i| index[i].len());

        let start_offset = index
            .iter()
            .zip(self.strides.iter())
            .map(|(range, stride)| *stride * range.start)
            .sum::<usize>();

        Self {
            offset: start_offset,
            shape,
            strides: self.strides,
            data_type: PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize; R] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize; R] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}
