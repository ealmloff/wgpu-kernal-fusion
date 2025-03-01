use wgpu_compute::*;

#[tokio::main]
async fn main() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });

    let tensor1 = Tensor::new(&device, &vec![[1., 2.]; 100]);
    let tensor2 = Tensor::new(&device, &vec![[1., 2.]; 100]);
    // This gets fused into a single kernel
    let new = Tensor::cat(
        [
            (((&tensor1 + &tensor2).cast::<half::f16>() + 1.) / 2.)
                .sum(1)
                .cast::<f32>()
                .slice([0..10])
                .silu()
                .silu()
                .silu()
                .silu(),
            tensor1.sum(1),
        ],
        0,
    );
    new.as_slice().await.unwrap();
}
