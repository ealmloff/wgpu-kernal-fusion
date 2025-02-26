use std::ops::Range;

pub(crate) const TILE_SIZE: u32 = 8;

fn continuous_strides(shape: &[usize]) -> Box<[usize]> {
    let mut acc = 1;
    let mut strides = vec![0; shape.len()].into_boxed_slice();
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

#[derive(Clone)]
pub struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Layout {
    pub fn contiguous(shape: &[usize]) -> Self {
        let strides = continuous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
        }
    }

    pub fn from_parts(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        Self {
            offset,
            shape,
            strides,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == continuous_strides(&self.shape)
    }

    pub fn slice(&self, index: &[Range<usize>]) -> Self {
        let shape = index.iter().map(|range| range.len()).collect();

        let start_offset = index
            .iter()
            .zip(self.strides.iter())
            .map(|(range, stride)| *stride * range.start)
            .sum::<usize>();

        Self {
            offset: start_offset,
            shape,
            strides: self.strides.clone(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[test]
fn test_contiguous() {
    let layout = Layout::contiguous(&[2, 3]);
    assert!(layout.is_contiguous());
    assert!(!layout.slice(&[0..1, 0..1]).is_contiguous());
    assert!(!layout.slice(&[1..2, 0..3]).is_contiguous());
}
