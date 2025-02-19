#![allow(unused)]
use criterion::BatchSize;
use futures::executor::block_on;
use wgpu_compute::{AddConst, ElementWiseFunction, ElementWiseOperation, MulConst, UnaryOp};
use wgpu_compute::{Device, MatMul, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 5] = [10, 100, 200, 500, 1000];

fn matmul(c: &mut Criterion) {
    // Here we have an async function to benchmark
    async fn matmul(device: Device, tensor_a: Tensor<2, f32>, tensor_b: Tensor<2, f32>) {
        let tensor = MatMul.run(&device, &tensor_a, &tensor_b).await;
        let _ = tensor.as_slice().await.unwrap();
    }

    {
        let mut group = c.benchmark_group("matmul-wgpu");

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

            group.bench_with_input(
                BenchmarkId::new("matmul-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_batched(
                        || {
                            (tensor.clone(), tensor.clone())
                        },
                        |(tensor_a, tensor_b)| matmul(device.clone(), tensor_a, tensor_b),
                        BatchSize::LargeInput,
                    );
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
