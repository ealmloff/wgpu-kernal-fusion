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

    let tensor1 = Tensor::new(&device, &[[1., 2.], [3., 4.], [5., 6.]]);
    let tensor2 = Tensor::new(&device, &[[1., 2.], [3., 4.], [5., 6.]]);
    // This gets fused into a single kernel
    let new: Tensor<1, half::f16> = (&tensor1 + &tensor2).sum(0).cast().silu();
    println!("{:?}", new.as_slice().await.unwrap());
}
