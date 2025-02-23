#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::BatchSize;
use futures::executor::block_on;
use ndarray::Axis;
use wgpu_compute::{Device, MatMul, Tensor};
use wgpu_compute::{
    ElementWiseFunction, ElementWiseOperation, PerformanceQueries, ReduceOperation, sum,
};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 4] = [100, 1000, 2000, 5000];

fn bench_sum_reduce(c: &mut Criterion) {
    async fn run_op(
        device: Device,
        tensor: Tensor<2, f32>,
        op: Arc<ReduceOperation<f32>>,
    ) -> Duration {
        let query = PerformanceQueries::new(&device);
        op.run_with_query(&tensor, 0, Some(&query));
        query.wait_for_results().await.elapsed()
    }

    {
        let mut group = c.benchmark_group("sum-wgpu");
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
                }
            });
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();
            let op = Arc::new(ReduceOperation::new(sum()));

            group.bench_with_input(BenchmarkId::new("sum-wgpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_custom(async |iters| {
                    let mut sum = Duration::ZERO;
                    while sum.is_zero() {
                        for _ in 0..iters {
                            let tensor = tensor.clone();
                            let op = op.clone();
                            sum += run_op(device.clone(), tensor, op).await;
                        }
                    }
                    sum
                })
            });
        }
    }

    {
        let mut group = c.benchmark_group("sum-ndarray");
        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("sum-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { tensor.sum_axis(Axis(0)) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, bench_sum_reduce);
criterion_main!(benches);
