use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Self {
        let shape = self.shape();
        assert!(start + length <= shape[axis]);
        assert!(axis < R);
        self.slice(std::array::from_fn(|i| {
            if i == axis {
                start..start + length
            } else {
                0..shape[i]
            }
        }))
    }
}
