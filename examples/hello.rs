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
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor1 = Tensor::new(&device, &data);
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor2 = Tensor::new(&device, &data);
    let new = tensor1.add(&tensor2).add_const(2.0);
    println!("{:?}", new.as_slice().await.unwrap());
}
