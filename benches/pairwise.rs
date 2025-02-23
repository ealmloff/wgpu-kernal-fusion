#![allow(unused)]
use std::sync::Arc;

use criterion::BatchSize;
use futures::executor::block_on;
use wgpu_compute::{add, ElementWiseFunction, ElementWiseOperation, PairWiseOperation};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn bench_add(c: &mut Criterion) {
    async fn run_op(
        device: Device,
        first: Tensor<2, f32>,
        second: Tensor<2, f32>,
        op: Arc<PairWiseOperation>,
    ) {
        op.run(&first, &second);
        let _ = second.as_slice().await.unwrap();
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
                b.to_async(FuturesExecutor).iter_batched(
                    || (tensor.clone(), tensor.clone(), op.clone()),
                    |(first, second, op)| run_op(device.clone(), first, second, op),
                    BatchSize::LargeInput,
                );
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
