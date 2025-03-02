#![allow(unused)]
use std::time::Duration;

use criterion::BatchSize;
use futures::executor::block_on;
use wgpu_compute::{Device, PerformanceQueries, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [100, 200, 500, 1000, 5000];

fn matmul(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("matmul-wgpu");

        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
                }
            });

            group.bench_with_input(
                BenchmarkId::new("matmul-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                _ = tensor.as_slice().await.unwrap();
                                let new = tensor.mat_mul(&tensor);
                                let timing = new.all_timing_information().await;
                                sum += timing.iter().map(|x| x.elapsed()).sum::<Duration>();
                            }
                        }
                        sum
                    });
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("matmul-ndarray");

        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("matmul-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || {
                            let matrix = ndarray::Array2::<f32>::ones((s, s));
                            (matrix.clone(), matrix.clone())
                        },
                        |(tensor_a, tensor_b)| async move { tensor_a.dot(&tensor_b) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, matmul);
criterion_main!(benches);
