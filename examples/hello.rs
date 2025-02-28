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

    let tensor1 = Tensor::new(&device, &vec![[1., 2.]; 10000]);
    let tensor2 = Tensor::new(&device, &vec![[1., 2.]; 10000]);
    // This gets fused into a single kernel
    let new = (((&tensor1 + &tensor2).cast::<half::f16>() + 1.) / 2.).sum(1).cast::<f32>().slice([0..10]).silu().silu().silu().silu();
    new.as_slice().await.unwrap();
}
