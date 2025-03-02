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

    let tensor = Tensor::new(&device, &vec![vec![[1.; 20]; 10]; 10]);
    let new = tensor.sum(0).sum(0).sum(0);
    let out: TensorSlice<0, f32> = new.as_slice().await.unwrap();
    println!("{:?}", out);
}
