use std::ops::Range;

use crate::{DataType, Tensor, TensorData, compute_graph::AnyComputeKey};

pub(crate) struct SliceOperation {
    pub(crate) input: AnyComputeKey,
    pub(crate) slice: Slice,
}

impl SliceOperation {
    pub fn new(input: AnyComputeKey, slice: Slice) -> Self {
        Self { input, slice }
    }

    pub fn run(&self, tensor: &TensorData) -> TensorData {
        tensor.slice(&self.slice.slices)
    }
}

pub(crate) struct Slice {
    pub(crate) slices: Box<[Range<usize>]>,
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn slice(&self, slice: [Range<usize>; R]) -> Tensor<R, T> {
        let slices = slice.into();

        self.add_slice(Slice { slices })
    }
}
