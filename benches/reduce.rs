#![allow(unused)]
use std::sync::Arc;

use criterion::BatchSize;
use futures::executor::block_on;
use ndarray::Axis;
use wgpu_compute::{sum, ElementWiseFunction, ElementWiseOperation, ReduceOperation};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn bench_sum_reduce(c: &mut Criterion) {
    async fn run_op(device: Device, tensor: Tensor<2, f32>, op: Arc<ReduceOperation>) {
        op.run(&tensor, 0);
        let _ = tensor.as_slice().await.unwrap();
    }

    {
        let mut group = c.benchmark_group("sum-wgpu");
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
            let op = Arc::new(ReduceOperation::new(sum()));

            group.bench_with_input(BenchmarkId::new("sum-wgpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_batched(
                    || (tensor.clone(), op.clone()),
                    |(tensor, op)| run_op(device.clone(), tensor, op),
                    BatchSize::LargeInput,
                );
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
