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
pub(crate) struct TensorLayout {
    pub(crate) data: Box<[u32]>,
}

impl TensorLayout {
    pub(crate) fn rank(&self) -> usize {
        (self.data.len() - 1) / 2
    }

    pub(crate) fn wgsl_type_definition(&self, kernel: &mut String) {
        let rank = self.rank();
        kernel.push_str("struct TensorLayout {\n");
        for i in 0..rank {
            kernel.push_str(&format!("\tstride_{}: u32,\n", i));
        }
        for i in 0..rank {
            kernel.push_str(&format!("\tshape_{}: u32,\n", i));
        }
        kernel.push_str("\toffset: u32,\n");
        kernel.push_str("}\n");
    }
}

impl From<&Layout> for TensorLayout {
    fn from(layout: &Layout) -> Self {
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
