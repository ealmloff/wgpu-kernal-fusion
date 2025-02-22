use std::{cell::OnceCell, fmt::Display};

use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use crate::{
    layout::{TensorLayout, TILE_SIZE},
    Tensor,
};

#[cfg(test)]
use crate::Device;

pub struct ElementWiseOperation {
    dtype: String,
    functions: Vec<ElementWiseFunction>,
    dense_kernel: OnceCell<wgpu::ShaderModule>,
    sparse_kernel: OnceCell<wgpu::ShaderModule>,
}

impl Default for ElementWiseOperation {
    fn default() -> Self {
        Self::new([])
    }
}

impl ElementWiseOperation {
    pub fn new(functions: impl IntoIterator<Item = ElementWiseFunction>) -> Self {
        Self {
            dtype: "f32".to_string(),
            functions: functions.into_iter().collect(),
            dense_kernel: OnceCell::new(),
            sparse_kernel: OnceCell::new(),
        }
    }

    pub fn then(self, other: ElementWiseFunction) -> Self {
        let mut functions = self.functions;
        functions.push(other);
        let dtype = self.dtype;
        Self {
            dtype,
            functions,
            dense_kernel: OnceCell::new(),
            sparse_kernel: OnceCell::new(),
        }
    }

    fn dtype(&self) -> String {
        self.dtype.clone()
    }

    fn modify_element(&self, inline: bool, index: &str, kernel: &mut String) {
        if !inline {
            let call = self
                .functions
                .iter()
                .fold(format!("matrix[{index}]"), |acc, f| f.call(acc));

            kernel.push_str(&format!("matrix[{index}] = {call};\n"));
        } else {
            kernel.push_str(&format!("var data = matrix[{index}];\n"));
            for function in &self.functions {
                kernel.push_str(&function.operation);
                kernel.push('\n');
            }
            kernel.push_str(&format!("matrix[{index}] = data;\n"));
        }
    }

    fn tiled_map<const R: usize>(&self, blocksize: u32, inline: bool, contiguous: bool) -> String {
        const {
            assert!(R <= 3, "TensorLayout only supports up to 3 rank tensors");
        }

        let dtype = &self.dtype;

        let mut kernel = String::new();
        TensorLayout::<R>::wgsl_type_definition(&mut kernel);
        kernel.push_str("@group(0) @binding(0) var<uniform> tensor_layout: TensorLayout;\n");
        kernel.push_str(&format!(
            "@group(0) @binding(1) var<storage, read_write> matrix: array<{dtype}>;\n"
        ));
        kernel.push_str(&format!("const BLOCKSIZE: u32 = {blocksize}u;\n"));
        kernel.push_str(&format!("const TILE_SIZE: u32 = {TILE_SIZE}u;\n"));
        if !inline {
            for function in &self.functions {
                kernel.push_str(&function.function(&self.dtype));
            }
        }
        kernel.push_str("\n@compute @workgroup_size(");
        if contiguous {
            kernel.push_str("BLOCKSIZE");
        } else {
            for i in 0..R {
                kernel.push_str(&format!("BLOCKSIZE"));
                if i < R - 1 {
                    kernel.push_str(", ");
                }
            }
        }
        kernel.push_str(")\n");
        kernel.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n");
        if contiguous {
            for local_index in 0..TILE_SIZE {
                let index = format!("index_{local_index}");
                kernel.push_str(
                    &format!("\t\tlet {index} = global_id.x * TILE_SIZE + {local_index} + tensor_layout.offset;\n"),
                );
                kernel.push_str(&format!("\t\tif {index} < \n"));
                for i in 0..R {
                    kernel.push_str(&format!("tensor_layout.shape_{i}"));
                    if i < R - 1 {
                        kernel.push_str(" * ");
                    }
                }
                kernel.push_str(" {\n");
                kernel.push_str("\t\t\t");
                self.modify_element(inline, &index, &mut kernel);
                kernel.push_str("\t\t}\n");
            }
        } else {
            for i in 0..R {
                let index = ["x", "y", "z"][i];
                kernel.push_str(&format!(
                    "\tlet tile_index_{i} = global_id.{index} * TILE_SIZE + tensor_layout.offset;\n"
                ));
            }
            kernel.push_str("\n");

            for i in 0..R {
                for _ in 0..(i + 1) {
                    kernel.push('\t');
                }
                kernel.push_str(&format!("for (var local_index_{i} = 0u; local_index_{i} < TILE_SIZE; local_index_{i}++) {{\n"));
            }

            for i in 0..R {
                for _ in 0..(R + 1) {
                    kernel.push('\t');
                }
                kernel.push_str(&format!(
                    "let merged_index_{i} = tile_index_{i} + local_index_{i};\n"
                ));
            }

            for _ in 0..(R + 1) {
                kernel.push('\t');
            }

            kernel.push_str("if ");
            for i in 0..R {
                kernel.push_str(&format!("merged_index_{i} < tensor_layout.shape_{i} && "));
            }
            kernel.push_str("true {\n");
            for _ in 0..(R + 2) {
                kernel.push('\t');
            }
            kernel.push_str(&format!("let index = "));
            if contiguous {
                kernel.push_str("global_id.x * TILE_SIZE;\n");
            } else {
                for i in 0..R {
                    kernel.push_str(&format!("tensor_layout.stride_{i} * merged_index_{i} + "));
                }
                kernel.push_str("0;\n");
            }
            for _ in 0..(R + 2) {
                kernel.push('\t');
            }
            self.modify_element(inline, "index", &mut kernel);

            for _ in 0..(R + 1) {
                kernel.push('\t');
            }
            kernel.push_str("}\n");

            for i in (0..R).rev() {
                for _ in 0..(i + 1) {
                    kernel.push('\t');
                }
                kernel.push_str("}\n");
            }
        }

        kernel.push_str("}\n");

        kernel
    }

    pub fn run<const R: usize>(&self, tensor: &Tensor<R, f32>) {
        let contiguous = tensor.layout().is_contiguous();
        let max_blocksize = if contiguous {
            256
        } else {
            // max_blocksize^R = 256
            (256f64.powf(1. / R as f64)).floor() as u32
        };
        let module = if contiguous {
            self.dense_kernel.get_or_init(|| {
                let source = self.tiled_map::<R>(max_blocksize, true, contiguous);
                tensor.device().create_shader_module(source)
            })
        } else {
            self.sparse_kernel.get_or_init(|| {
                let source = self.tiled_map::<R>(max_blocksize, true, contiguous);
                tensor.device().create_shader_module(source)
            })
        };

        let bind_group_layout = tensor.device().wgpu_device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );
        let compute_pipeline_layout =
            tensor
                .device()
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = tensor.device().wgpu_device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compute_pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
            },
        );

        let layout = TensorLayout::for_tensor(tensor);

        let layout =
            tensor
                .device()
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&layout.data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bind_group =
            tensor
                .device()
                .wgpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: layout.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: tensor.buffer().as_entire_binding(),
                        },
                    ],
                });

        let mut encoder = tensor
            .device()
            .wgpu_device()
            .create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let layout = tensor.layout();
            let shape = layout.shape();
            let (workgroup_size_x, workgroup_size_y, workgroup_size_z) = if contiguous {
                (
                    shape
                        .iter()
                        .map(|x| *x as u32)
                        .product::<u32>()
                        .div_ceil(TILE_SIZE * max_blocksize),
                    1,
                    1,
                )
            } else {
                let workgroup_size_x = shape
                    .get(0)
                    .map(|x| (*x as u32).div_ceil(TILE_SIZE * max_blocksize))
                    .unwrap_or(1);
                let workgroup_size_y = shape
                    .get(1)
                    .map(|x| (*x as u32).div_ceil(TILE_SIZE * max_blocksize))
                    .unwrap_or(1);
                let workgroup_size_z = shape
                    .get(2)
                    .map(|x| (*x as u32).div_ceil(TILE_SIZE * max_blocksize))
                    .unwrap_or(1);
                (workgroup_size_x, workgroup_size_y, workgroup_size_z)
            };
            cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z)
        }
        tensor.device().wgpu_queue().submit(Some(encoder.finish()));
    }
}

pub struct ElementWiseFunction {
    name_id: u64,
    operation: String,
}

impl<Op: UnaryOp> From<Op> for ElementWiseFunction {
    fn from(op: Op) -> Self {
        Self::new(op.source())
    }
}

impl ElementWiseFunction {
    fn new(operation: impl Display) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            name_id,
            operation: operation.to_string(),
        }
    }

    fn call(&self, data: impl Display) -> String {
        let name_id = self.name_id;
        format!("unary_{name_id}({data})")
    }

    fn function(&self, dtype: &str) -> String {
        let Self { name_id, operation } = self;
        format!(
            r#"fn unary_{name_id}(input: {dtype}) -> {dtype} {{
    var data = input;
    {operation}
    return data;
}}"#
        )
    }
}

pub trait UnaryOp {
    fn source(&self) -> impl Display;
    fn run<const R: usize>(&self, tensor: &Tensor<R, f32>) {
        ElementWiseOperation::new([ElementWiseFunction::new(self.source())]).run(tensor);
    }
}

pub struct AddConst(f32);

impl AddConst {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl UnaryOp for AddConst {
    fn source(&self) -> impl Display {
        format!("data = data + {};", self.0)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    AddConst::new(1.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    AddConst::new(1.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 3.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 5.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 7.);

    let data = [1., 2.];
    let tensor = Tensor::new(&device, &data);

    AddConst::new(1.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 3.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_sliced() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let sliced = tensor.slice([0..3, 0..1]);

    AddConst::new(1.0).run(&sliced);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 2.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 4.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_large() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    const BUF_SIZE: usize = 0x01000000;
    let data = vec![10.; BUF_SIZE];
    let tensor = Tensor::new(&device, &data);

    AddConst::new(1.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    for i in 0..BUF_SIZE {
        assert_eq!(output[[i]], 11.0);
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_merge_add_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    ElementWiseOperation::default()
        .then(ElementWiseFunction::new(AddConst::new(1.0).source()))
        .then(ElementWiseFunction::new(MulConst::new(2.0).source()))
        .run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 4.);
    assert_eq!(output[[0, 1]], 6.);
    assert_eq!(output[[1, 0]], 8.);
    assert_eq!(output[[1, 1]], 10.);
    assert_eq!(output[[2, 0]], 12.);
    assert_eq!(output[[2, 1]], 14.);
}

pub struct SubConst(f32);

impl SubConst {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl UnaryOp for SubConst {
    fn source(&self) -> impl Display {
        format!("data = data - {};", self.0)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sub_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    SubConst::new(1.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 0.);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 2.);
    assert_eq!(output[[1, 1]], 3.);
    assert_eq!(output[[2, 0]], 4.);
    assert_eq!(output[[2, 1]], 5.);
}

pub struct MulConst(f32);

impl MulConst {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl UnaryOp for MulConst {
    fn source(&self) -> impl Display {
        format!("data = data * {};", self.0)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mul_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    MulConst::new(2.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 6.);
    assert_eq!(output[[1, 1]], 8.);
    assert_eq!(output[[2, 0]], 10.);
    assert_eq!(output[[2, 1]], 12.);
}

pub struct DivConst(f32);

impl DivConst {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl UnaryOp for DivConst {
    fn source(&self) -> impl Display {
        format!("data = data / {};", self.0)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_div_const() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    DivConst::new(2.0).run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0, 0]], 0.5);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 1.5);
    assert_eq!(output[[1, 1]], 2.);
    assert_eq!(output[[2, 0]], 2.5);
    assert_eq!(output[[2, 1]], 3.);
}

pub struct Exp;

impl Exp {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Exp {
    fn source(&self) -> impl Display {
        format!("data = exp(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Exp::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].exp()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp()).abs() < 0.001);
}

pub struct Exp2;

impl Exp2 {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Exp2 {
    fn source(&self) -> impl Display {
        format!("data = exp2(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp2() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Exp2::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].exp2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp2()).abs() < 0.001);
}

pub struct Log;

impl Log {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Log {
    fn source(&self) -> impl Display {
        format!("data = log(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Log::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].ln()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].ln()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].ln()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].ln()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].ln()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].ln()).abs() < 0.001);
}

pub struct Log2;

impl Log2 {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Log2 {
    fn source(&self) -> impl Display {
        format!("data = log2(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log2() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Log2::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].log2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].log2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].log2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].log2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].log2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].log2()).abs() < 0.001);
}

pub struct Sqrt;

impl Sqrt {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Sqrt {
    fn source(&self) -> impl Display {
        format!("data = sqrt(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sqrt() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Sqrt::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sqrt()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sqrt()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sqrt()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sqrt()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sqrt()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sqrt()).abs() < 0.001);
}

pub struct Sin;

impl Sin {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Sin {
    fn source(&self) -> impl Display {
        format!("data = sin(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sin() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Sin::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sin()).abs() < 0.001);
}

pub struct Cos;

impl Cos {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Cos {
    fn source(&self) -> impl Display {
        format!("data = cos(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cos() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Cos::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].cos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cos()).abs() < 0.001);
}

pub struct Tan;

impl Tan {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Tan {
    fn source(&self) -> impl Display {
        format!("data = tan(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tan() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Tan::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].tan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tan()).abs() < 0.001);
}

pub struct Asin;

impl Asin {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Asin {
    fn source(&self) -> impl Display {
        format!("data = asin(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asin() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [
        [1.0f32.sin(), 2.0f32.sin()],
        [3.0f32.sin(), 4.0f32.sin()],
        [5.0f32.sin(), 6.0f32.sin()],
    ];
    let tensor = Tensor::new(&device, &data);

    Asin::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].asin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asin()).abs() < 0.001);
}

pub struct Acos;

impl Acos {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Acos {
    fn source(&self) -> impl Display {
        format!("data = acos(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acos() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [
        [1.0f32.cos(), 2.0f32.cos()],
        [3.0f32.cos(), 4.0f32.cos()],
        [5.0f32.cos(), 6.0f32.cos()],
    ];
    let tensor = Tensor::new(&device, &data);

    Acos::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].acos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acos()).abs() < 0.001);
}

pub struct Atan;

impl Atan {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Atan {
    fn source(&self) -> impl Display {
        format!("data = atan(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atan() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1. / 1., 1. / 2.], [1. / 3., 1. / 4.], [1. / 5., 1. / 6.]];
    let tensor = Tensor::new(&device, &data);

    Atan::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].atan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atan()).abs() < 0.001);
}

pub struct Sinh;

impl Sinh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Sinh {
    fn source(&self) -> impl Display {
        format!("data = sinh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sinh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Sinh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].sinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sinh()).abs() < 0.001);
}

pub struct Cosh;

impl Cosh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Cosh {
    fn source(&self) -> impl Display {
        format!("data = cosh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cosh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Cosh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].cosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cosh()).abs() < 0.001);
}

pub struct Tanh;

impl Tanh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Tanh {
    fn source(&self) -> impl Display {
        format!("data = tanh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tanh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    Tanh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].tanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tanh()).abs() < 0.001);
}

pub struct Asinh;

impl Asinh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Asinh {
    fn source(&self) -> impl Display {
        format!("data = asinh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asinh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [
        [1.0f32.sinh(), 2.0f32.sinh()],
        [3.0f32.sinh(), 4.0f32.sinh()],
        [5.0f32.sinh(), 6.0f32.sinh()],
    ];
    let tensor = Tensor::new(&device, &data);

    Asinh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].asinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asinh()).abs() < 0.001);
}

pub struct Acosh;

impl Acosh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Acosh {
    fn source(&self) -> impl Display {
        format!("data = acosh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acosh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [
        [1.0f32.cosh(), 2.0f32.cosh()],
        [3.0f32.cosh(), 4.0f32.cosh()],
        [5.0f32.cosh(), 6.0f32.cosh()],
    ];
    let tensor = Tensor::new(&device, &data);

    Acosh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].acosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acosh()).abs() < 0.001);
}

pub struct Atanh;

impl Atanh {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Atanh {
    fn source(&self) -> impl Display {
        format!("data = atanh(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atanh() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [
        [1.0f32.tanh(), 2.0f32.tanh()],
        [3.0f32.tanh(), 4.0f32.tanh()],
        [5.0f32.tanh(), 6.0f32.tanh()],
    ];
    let tensor = Tensor::new(&device, &data);

    Atanh::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].atanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atanh()).abs() < 0.001);
}

pub struct Abs;

impl Abs {
    pub const fn new() -> Self {
        Self
    }
}

impl UnaryOp for Abs {
    fn source(&self) -> impl Display {
        format!("data = abs(data);")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_abs() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    Abs::new().run(&tensor);

    let output = tensor.as_slice().await.unwrap();
    println!("{:?}", output);
    assert!((output[[0, 0]] - data[0][0].abs()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].abs()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].abs()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].abs()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].abs()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].abs()).abs() < 0.001);
}
