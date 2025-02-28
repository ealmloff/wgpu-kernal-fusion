use std::{
    collections::HashMap,
    sync::{Arc, RwLock, atomic::AtomicUsize},
};

use arc_swap::ArcSwap;

mod layout_pass;
mod resolve;
mod visit;
mod visualize;

use crate::{
    Device, ElementWiseOperation, MatMulOperation, PairWiseOperation, ReduceOperation,
    slice::SliceOperation, tensor::TensorData,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ElementWiseComputeNodeKey(usize);
impl ElementWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PairWiseComputeNodeKey(usize);
impl PairWiseComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatMulComputeNodeKey(usize);
impl MatMulComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ReduceComputeNodeKey(usize);
impl ReduceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SliceComputeNodeKey(usize);
impl SliceComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct TensorComputeNodeKey(usize);
impl TensorComputeNodeKey {
    fn new() -> Self {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        Self(COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AnyComputeKey {
    ElementWiseComputeNodeKey(ElementWiseComputeNodeKey),
    PairWiseComputeNodeKey(PairWiseComputeNodeKey),
    MatMulComputeNodeKey(MatMulComputeNodeKey),
    ReduceComputeNodeKey(ReduceComputeNodeKey),
    SliceComputeNodeKey(SliceComputeNodeKey),
    TensorComputeNodeKey(TensorComputeNodeKey),
}

impl From<ElementWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: ElementWiseComputeNodeKey) -> Self {
        Self::ElementWiseComputeNodeKey(value)
    }
}

impl From<PairWiseComputeNodeKey> for AnyComputeKey {
    fn from(value: PairWiseComputeNodeKey) -> Self {
        Self::PairWiseComputeNodeKey(value)
    }
}

impl From<MatMulComputeNodeKey> for AnyComputeKey {
    fn from(value: MatMulComputeNodeKey) -> Self {
        Self::MatMulComputeNodeKey(value)
    }
}

impl From<ReduceComputeNodeKey> for AnyComputeKey {
    fn from(value: ReduceComputeNodeKey) -> Self {
        Self::ReduceComputeNodeKey(value)
    }
}

impl From<TensorComputeNodeKey> for AnyComputeKey {
    fn from(value: TensorComputeNodeKey) -> Self {
        Self::TensorComputeNodeKey(value)
    }
}

impl From<SliceComputeNodeKey> for AnyComputeKey {
    fn from(value: SliceComputeNodeKey) -> Self {
        Self::SliceComputeNodeKey(value)
    }
}

#[derive(Clone, Default)]
pub(crate) struct ComputeGraph {
    inner: Arc<ArcSwap<RwLock<ComputeGraphInner>>>,
}

impl ComputeGraph {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let write = self.inner.load();
        let mut inner = write.write().unwrap();
        f(&mut inner)
    }

    pub(crate) fn merge(&self, other: &Self) {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return;
        }
        self.with_mut(|inner| {
            other.with_mut(|other_inner| {
                inner.element_wise.extend(other_inner.element_wise.drain());
                inner.pair_wise.extend(other_inner.pair_wise.drain());
                inner.mat_mul.extend(other_inner.mat_mul.drain());
                inner.reduce.extend(other_inner.reduce.drain());
                inner.slice.extend(other_inner.slice.drain());
                inner.tensor.extend(other_inner.tensor.drain());
            })
        });
        other.inner.store(self.inner.load_full());
    }

    pub(crate) fn create_element_wise(
        &self,
        function: ElementWiseOperation,
    ) -> ElementWiseComputeNodeKey {
        let id = ElementWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.element_wise.insert(id, function));
        id
    }

    pub(crate) fn create_pair_wise(&self, function: PairWiseOperation) -> PairWiseComputeNodeKey {
        let id = PairWiseComputeNodeKey::new();
        self.with_mut(|inner| inner.pair_wise.insert(id, function));
        id
    }

    pub(crate) fn create_mat_mul(&self, function: MatMulOperation) -> MatMulComputeNodeKey {
        let id = MatMulComputeNodeKey::new();
        self.with_mut(|inner| inner.mat_mul.insert(id, function));
        id
    }

    pub(crate) fn create_reduce(&self, function: ReduceOperation) -> ReduceComputeNodeKey {
        let id = ReduceComputeNodeKey::new();
        self.with_mut(|inner| inner.reduce.insert(id, function));
        id
    }

    pub(crate) fn create_slice(&self, op: SliceOperation) -> SliceComputeNodeKey {
        let id = SliceComputeNodeKey::new();
        self.with_mut(|inner| inner.slice.insert(id, op));
        id
    }

    pub(crate) fn create_tensor(&self, info: TensorData) -> TensorComputeNodeKey {
        let id = TensorComputeNodeKey::new();
        self.with_mut(|inner| inner.tensor.insert(id, info));
        id
    }

    pub(crate) fn resolve(&self, key: AnyComputeKey, device: &Device) -> TensorData {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&Default::default());
        let data = self.with_mut(|inner| inner.resolve(key, &mut encoder));
        device.wgpu_queue().submit(Some(encoder.finish()));
        data
    }
}

#[derive(Default)]
struct ComputeGraphInner {
    element_wise: HashMap<ElementWiseComputeNodeKey, ElementWiseOperation>,
    pair_wise: HashMap<PairWiseComputeNodeKey, PairWiseOperation>,
    mat_mul: HashMap<MatMulComputeNodeKey, MatMulOperation>,
    reduce: HashMap<ReduceComputeNodeKey, ReduceOperation>,
    slice: HashMap<SliceComputeNodeKey, SliceOperation>,
    tensor: HashMap<TensorComputeNodeKey, TensorData>,
}
