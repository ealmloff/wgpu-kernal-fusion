#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, BatchSize};
use futures::executor::block_on;
use wgpu_compute::{add_const, mul_const, ElementWiseFunction, ElementWiseOperation, PerformanceQueries};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn fused(c: &mut Criterion) {
    async fn run_op(device: Device, tensor: Tensor<2, f32>, op: Arc<ElementWiseOperation>) -> Duration {
        let query = PerformanceQueries::new(&device);
        op.run_with_query(&tensor, Some(&query));
        query.wait_for_results().await.elapsed()
    }

    async fn add_const_separate(
        device: Device,
        tensor: Tensor<2, f32>,
        op1: Arc<ElementWiseOperation>,
        op2: Arc<ElementWiseOperation>,
    ) -> Duration {
        let query = PerformanceQueries::new(&device);
        op1.run_with_query(&tensor, Some(&query));
        op2.run_with_query(&tensor, Some(&query));
        query.wait_for_results().await.elapsed()
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
                .then(add_const(1.0))
                .then(add_const(1.0));
            let op = Arc::new(fused);

            group.bench_with_input(
                BenchmarkId::new("add-const-fused-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        for _ in 0..iters {
                            let tensor = tensor.clone();
                            let op = op.clone();
                            sum += run_op(device.clone(), tensor, op).await; 
                        }
                        sum
                    })
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

            let op1 = Arc::new(ElementWiseOperation::default().then(add_const(1.0)));
            let op2 = Arc::new(ElementWiseOperation::default().then(add_const(1.0)));

            group.bench_with_input(
                BenchmarkId::new("add-const-separate-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        for _ in 0..iters {
                            sum += add_const_separate(device.clone(), tensor.clone(), op1.clone(), op2.clone()).await;
                        }
                        sum
                    })
                },
            );
        }
    }
}

criterion_group!(benches, fused);
criterion_main!(benches);
