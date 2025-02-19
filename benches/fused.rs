#![allow(unused)]
use std::sync::Arc;

use criterion::BatchSize;
use futures::executor::block_on;
use wgpu_compute::{AddConst, ElementWiseFunction, ElementWiseOperation, MulConst, UnaryOp};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn fused(c: &mut Criterion) {
    async fn run_op(device: Device, tensor: Tensor<2, f32>, op: Arc<ElementWiseOperation>) {
        op.run(&tensor);
        let _ = tensor.as_slice().await.unwrap();
    }

    async fn add_const_separate(device: Device, tensor: Tensor<2, f32>, op1: Arc<ElementWiseOperation>, op2: Arc<ElementWiseOperation>) {
        op1.run(&tensor);
        op2.run(&tensor);
        let _ = tensor.as_slice().await.unwrap();
    }

    {
        let mut group = c.benchmark_group("add-const-fused-wgpu");
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::Maintain::Wait);
                }
            });
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            let fused = ElementWiseOperation::default()
            .then(ElementWiseFunction::from(AddConst::new(1.0)))
            .then(ElementWiseFunction::from(AddConst::new(1.0)));
        let op = Arc::new(fused);

            group.bench_with_input(
                BenchmarkId::new("add-const-fused-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_batched(
                        || (tensor.clone(), op.clone()),
                        |(tensor, op)| run_op(device.clone(), tensor, op.clone()),
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("add-const-separate-wgpu");
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::Maintain::Wait);
                }
            });
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            let op1 = Arc::new(ElementWiseOperation::default().then(ElementWiseFunction::from(AddConst::new(1.0))));
            let op2 = Arc::new(ElementWiseOperation::default().then(ElementWiseFunction::from(AddConst::new(1.0))));

            group.bench_with_input(
                BenchmarkId::new("add-const-separate-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_batched(
                        || (tensor.clone(), op1.clone(), op2.clone()),
                        |(tensor, op1, op2)| add_const_separate(device.clone(), tensor, op1, op2),
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, fused);
criterion_main!(benches);
