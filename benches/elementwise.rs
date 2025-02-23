#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, BatchSize};
use futures::executor::block_on;
use wgpu_compute::{
    add_const, mul_const, ElementWiseFunction, ElementWiseOperation, PerformanceQueries,
};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn bench_add_const(c: &mut Criterion) {
    async fn run_op(
        device: Device,
        tensor: Tensor<2, f32>,
        op: Arc<ElementWiseOperation>,
        query: &PerformanceQueries,
    ) {
        op.run_with_query(&tensor, Some(query));
        let _ = tensor.as_slice().await.unwrap();
    }

    {
        let mut group = c.benchmark_group("add-const-wgpu");
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
            let op = Arc::new(ElementWiseOperation::default().then(add_const(1.0)));

            group.bench_with_input(
                BenchmarkId::new("add-const-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        for _ in 0..iters {
                            let tensor = tensor.clone();
                            let op = op.clone();
                            let query = PerformanceQueries::new(&device);
                            black_box(run_op(device.clone(), tensor, op, &query).await);
                            sum += query.wait_for_results().await.elapsed();
                        }
                        sum
                    })
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("add-const-ndarray");
        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("add-const-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { tensor.map(|x| x + 1.) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, bench_add_const);
criterion_main!(benches);
