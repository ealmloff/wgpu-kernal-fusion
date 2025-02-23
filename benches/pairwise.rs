#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::BatchSize;
use futures::executor::block_on;
use wgpu_compute::{Device, MatMul, Tensor};
use wgpu_compute::{
    ElementWiseFunction, ElementWiseOperation, PairWiseOperation, PerformanceQueries, add,
};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 4] = [100, 1000, 2000, 5000];

fn bench_add(c: &mut Criterion) {
    async fn run_op(
        device: Device,
        first: Tensor<2, f32>,
        second: Tensor<2, f32>,
        op: Arc<PairWiseOperation>,
    ) -> Duration {
        let query = PerformanceQueries::new(&device);
        op.run_with_query(&first, &second, Some(&query));
        query.wait_for_results().await.elapsed()
    }

    {
        let mut group = c.benchmark_group("add-wgpu");
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
            let op = Arc::new(PairWiseOperation::new(add()));

            group.bench_with_input(BenchmarkId::new("add-wgpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_custom(async |iters| {
                    let mut sum = Duration::ZERO;
                    while sum.is_zero() {
                        for _ in 0..iters {
                            sum +=
                                run_op(device.clone(), tensor.clone(), tensor.clone(), op.clone())
                                    .await;
                        }
                    }
                    sum
                })
            });
        }
    }

    {
        let mut group = c.benchmark_group("add-ndarray");
        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("add-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { &tensor + &tensor },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, bench_add);
criterion_main!(benches);
