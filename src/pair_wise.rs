use std::{fmt::Display, marker::PhantomData, sync::OnceLock};

use wgpu::{PipelineCompilationOptions, util::DeviceExt};

use crate::{
    ElementWiseOperation, Tensor,
    layout::{TILE_SIZE, TensorLayout},
    query::PerformanceQueries,
    tensor::DataType,
};

pub struct PairWiseOperation<T> {
    pre_element_wise: [ElementWiseOperation<T>; 2],
    function: PairWiseFunction,
    post_element_wise: ElementWiseOperation<T>,
    dense_kernel: OnceLock<wgpu::ShaderModule>,
    sparse_kernel: OnceLock<wgpu::ShaderModule>,
    datatype: PhantomData<T>,
}

impl<T: DataType> PairWiseOperation<T> {
    pub fn new(function: PairWiseFunction) -> Self {
        Self {
            pre_element_wise: [
                ElementWiseOperation::default(),
                ElementWiseOperation::default(),
            ],
            function,
            post_element_wise: ElementWiseOperation::default(),
            dense_kernel: OnceLock::new(),
            sparse_kernel: OnceLock::new(),
            datatype: PhantomData,
        }
    }

    pub(crate) fn modify_data(&self, inline: bool, kernel: &mut String) {
        for (operation, input) in self.pre_element_wise.iter().zip(["a", "b"]) {
            if !operation.is_empty() {
                kernel.push_str("{\n");
                kernel.push_str(&format!("\tvar data = {input};\n"));
                kernel.push('\t');
                operation.modify_data(inline, &mut *kernel);
                kernel.push_str(&format!("\t{input} = data;\n"));
                kernel.push_str("}\n");
            }
        }
        if !inline {
            let call = self.function.call("a", "b");
            kernel.push_str(&format!("data = {call};\n"));
        } else {
            kernel.push_str(&self.function.operation);
            kernel.push('\n');
        }
        self.post_element_wise.modify_data(inline, &mut *kernel);
    }

    pub(crate) fn add_functions(&self, inline: bool, kernel: &mut String) {
        for operation in self.pre_element_wise.iter() {
            operation.add_functions(inline, &mut *kernel);
        }
        if !inline {
            kernel.push_str(&self.function.function(T::WGSL_TYPE));
        }
        self.post_element_wise.add_functions(inline, &mut *kernel);
    }

    fn tiled_map<const R: usize>(&self, blocksize: u32, inline: bool, contiguous: bool) -> String {
        const {
            assert!(R <= 3, "TensorLayout only supports up to 3 rank tensors");
        }

        let dtype = T::WGSL_TYPE;

        let mut kernel = String::new();
        if dtype == "f16" {
            kernel.push_str("enable f16;\n");
        }
        TensorLayout::<R>::wgsl_type_definition(&mut kernel);
        kernel.push_str("@group(0) @binding(0) var<uniform> first_tensor_layout: TensorLayout;\n");
        kernel.push_str(&format!(
            "@group(0) @binding(1) var<storage, read_write> first_tensor: array<{dtype}>;\n"
        ));
        kernel.push_str("@group(0) @binding(2) var<uniform> out_tensor_layout: TensorLayout;\n");
        kernel.push_str(&format!(
            "@group(0) @binding(3) var<storage, read_write> out_tensor: array<{dtype}>;\n"
        ));
        kernel.push_str(&format!("const BLOCKSIZE: u32 = {blocksize}u;\n"));
        kernel.push_str(&format!("const TILE_SIZE: u32 = {TILE_SIZE}u;\n"));
        self.add_functions(inline, &mut kernel);
        kernel.push_str("\n@compute @workgroup_size(");
        if contiguous {
            kernel.push_str("BLOCKSIZE");
        } else {
            for i in 0..R {
                kernel.push_str("BLOCKSIZE");
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
                kernel.push_str(&format!(
                    "\t\tlet {index} = global_id.x * TILE_SIZE + {local_index}u;\n"
                ));
                kernel.push_str(&format!("\t\tif {index} < \n"));
                for i in 0..R {
                    kernel.push_str(&format!("first_tensor_layout.shape_{i}"));
                    if i < R - 1 {
                        kernel.push_str(" * ");
                    }
                }
                kernel.push_str(" {\n");
                kernel.push_str(&format!("\t\t\tvar a = first_tensor[{index}];\n"));
                kernel.push_str(&format!("\t\t\tvar b = out_tensor[{index}];\n"));
                kernel.push_str(&format!("\t\t\tvar data: {dtype};\n"));
                kernel.push_str("\t\t\t");
                self.modify_data(inline, &mut kernel);
                kernel.push_str(&format!("\t\t\tout_tensor[{index}] = data;\n"));
                kernel.push_str("\t\t}\n");
            }
        } else {
            for i in 0..R {
                let index = ["x", "y", "z"][i];
                kernel.push_str(&format!(
                    "\tlet tile_index_{i} = global_id.{index} * TILE_SIZE;\n"
                ));
            }
            kernel.push('\n');

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
                kernel.push_str(&format!(
                    "merged_index_{i} < first_tensor_layout.shape_{i} && "
                ));
            }
            kernel.push_str("true {\n");
            for _ in 0..(R + 2) {
                kernel.push('\t');
            }
            for tensor_prefix in ["first", "out"].iter() {
                kernel.push_str(&format!(
                    "let {tensor_prefix}_index = {tensor_prefix}_tensor_layout.offset + "
                ));
                for i in 0..R {
                    kernel.push_str(&format!(
                        "{tensor_prefix}_tensor_layout.stride_{i} * merged_index_{i} + "
                    ));
                }
                kernel.push_str("0u;\n");
                for _ in 0..(R + 2) {
                    kernel.push('\t');
                }
            }
            kernel.push_str("\t\t\tvar a = first_tensor[first_index];\n");
            kernel.push_str(&format!("\t\t\tvar b = out_tensor[out_index];\n"));
            kernel.push_str(&format!("\t\t\tvar data: {dtype};\n"));
            kernel.push_str("\t\t\t");
            self.modify_data(inline, &mut kernel);
            kernel.push_str("\t\t\tout_tensor[out_index] = data;\n");

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

    pub fn run<const R: usize>(&self, first: &Tensor<R, T>, out: &Tensor<R, T>) {
        self.run_with_query(first, out, None);
    }

    pub fn run_with_query<const R: usize>(
        &self,
        first: &Tensor<R, T>,
        out: &Tensor<R, T>,
        query: Option<&PerformanceQueries>,
    ) {
        assert_eq!(first.layout().shape(), out.layout().shape());
        let contiguous = first.layout().is_contiguous() && out.layout().is_contiguous();
        let max_blocksize = if contiguous {
            256
        } else {
            // max_blocksize^R = 256
            (256f64.powf(1. / R as f64)).floor() as u32
        };
        let device = first.device();
        let module = if contiguous {
            self.dense_kernel.get_or_init(|| {
                let source = self.tiled_map::<R>(max_blocksize, true, contiguous);
                device.create_shader_module(source)
            })
        } else {
            self.sparse_kernel.get_or_init(|| {
                let source = self.tiled_map::<R>(max_blocksize, true, contiguous);
                device.create_shader_module(source)
            })
        };

        let bind_group_layout =
            device
                .wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let compute_pipeline_layout =
            device
                .wgpu_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_pipeline_layout),
                    module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: PipelineCompilationOptions::default(),
                });

        let first_layout = TensorLayout::for_tensor(first);
        let out_layout = TensorLayout::for_tensor(out);

        let first_layout =
            device
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&first_layout.data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let out_layout =
            device
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&out_layout.data),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bind_group = device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: first_layout.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: first.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_layout.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.buffer().as_entire_binding(),
                    },
                ],
            });

        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: query.map(|query| query.compute_timestamp_writes()),
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let layout = first.layout();
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
                    .first()
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
        if let Some(query) = query {
            query.resolve(&mut encoder);
        }
        device.wgpu_queue().submit(Some(encoder.finish()));
    }
}

#[derive(Clone)]
pub struct PairWiseFunction {
    name_id: u64,
    operation: String,
}

impl PairWiseFunction {
    fn new(operation: impl Display) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            name_id,
            operation: operation.to_string(),
        }
    }

    fn call(&self, a: impl Display, b: impl Display) -> String {
        let name_id = self.name_id;
        format!("binary_{name_id}({a}, {b})")
    }

    fn function(&self, dtype: &str) -> String {
        let Self { name_id, operation } = self;
        format!(
            r#"fn binary_{name_id}(a: {dtype}, b: {dtype}) -> {dtype} {{
    var data: {dtype};
    {operation}
    return data;
}}"#
        )
    }

    pub fn run<const R: usize, T: DataType>(&self, first: &Tensor<R, T>, out: &Tensor<R, T>) {
        self.run_with_query(first, out, None);
    }

    pub fn run_with_query<const R: usize, T: DataType>(
        &self,
        first: &Tensor<R, T>,
        out: &Tensor<R, T>,
        query: Option<&PerformanceQueries>,
    ) {
        PairWiseOperation::new(self.clone()).run_with_query(first, out, query);
    }
}

pub fn add() -> PairWiseFunction {
    PairWiseFunction::new(format!("data = a + b;"))
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let query = PerformanceQueries::new(&device);

    PairWiseOperation::new(add()).run_with_query(&tensor_a, &tensor_b, Some(&query));
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    println!("{}", query.wait_for_results().await);

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[1, 1]], 4. + 4.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
    assert_eq!(as_slice[[2, 1]], 6. + 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_f16() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let data_b = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let query = PerformanceQueries::new(&device);

    PairWiseOperation::new(add()).run_with_query(&tensor_a, &tensor_b, Some(&query));
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    println!("{}", query.wait_for_results().await);

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1. + 1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2. + 2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3. + 3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(4. + 4.));
    assert_eq!(as_slice[[2, 0]], half::f16::from_f32(5. + 5.));
    assert_eq!(as_slice[[2, 1]], half::f16::from_f32(6. + 6.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_const_mul_const_add_fused() {
    use crate::{Device, add_const, mul_const};

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let mut op = PairWiseOperation::new(add());
    op.pre_element_wise[0] = ElementWiseOperation::new([add_const(1.)]);
    op.pre_element_wise[1] = ElementWiseOperation::new([mul_const(2.)]);
    op.run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], (1. + 1.) + (1. * 2.));
    assert_eq!(as_slice[[0, 1]], (2. + 1.) + (2. * 2.));
    assert_eq!(as_slice[[1, 0]], (3. + 1.) + (3. * 2.));
    assert_eq!(as_slice[[1, 1]], (4. + 1.) + (4. * 2.));
    assert_eq!(as_slice[[2, 0]], (5. + 1.) + (5. * 2.));
    assert_eq!(as_slice[[2, 1]], (6. + 1.) + (6. * 2.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sub_const_fused() {
    use crate::{Device, sub_const};

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let mut op = PairWiseOperation::new(add());
    op.post_element_wise = ElementWiseOperation::new([sub_const(1.)]);
    op.run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. + 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2. - 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3. - 1.);
    assert_eq!(as_slice[[1, 1]], 4. + 4. - 1.);
    assert_eq!(as_slice[[2, 0]], 5. + 5. - 1.);
    assert_eq!(as_slice[[2, 1]], 6. + 6. - 1.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sparse() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_a = tensor_a.slice([0..3, 0..1]);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor_b = tensor_b.slice([0..3, 0..1]);

    PairWiseOperation::new(add()).run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
}

pub fn sub() -> PairWiseFunction {
    PairWiseFunction::new(format!("data = a - b;"))
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_sub() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    PairWiseOperation::new(sub()).run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. - 2.);
    assert_eq!(as_slice[[1, 0]], 3. - 3.);
    assert_eq!(as_slice[[1, 1]], 4. - 4.);
    assert_eq!(as_slice[[2, 0]], 5. - 5.);
    assert_eq!(as_slice[[2, 1]], 6. - 6.);
}

pub fn mul() -> PairWiseFunction {
    PairWiseFunction::new(format!("data = a * b;").to_string())
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_mul() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    PairWiseOperation::new(mul()).run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. * 1.);
    assert_eq!(as_slice[[0, 1]], 2. * 2.);
    assert_eq!(as_slice[[1, 0]], 3. * 3.);
    assert_eq!(as_slice[[1, 1]], 4. * 4.);
    assert_eq!(as_slice[[2, 0]], 5. * 5.);
    assert_eq!(as_slice[[2, 1]], 6. * 6.);
}

pub fn div() -> PairWiseFunction {
    PairWiseFunction::new(format!("data = a / b;").to_string())
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_div() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    PairWiseOperation::new(div()).run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1. / 1.);
    assert_eq!(as_slice[[0, 1]], 2. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 3.);
    assert_eq!(as_slice[[1, 1]], 4. / 4.);
    assert_eq!(as_slice[[2, 0]], 5. / 5.);
    assert_eq!(as_slice[[2, 1]], 6. / 6.);
}

pub fn pow() -> PairWiseFunction {
    PairWiseFunction::new(format!("data = pow(a, b);").to_string())
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_pow() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    PairWiseOperation::new(pow()).run(&tensor_a, &tensor_b);
    let as_slice = tensor_b.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert!((as_slice[[0, 0]] - 1_f32.powf(1.)) < 0.001);
    assert!((as_slice[[0, 1]] - 2_f32.powf(2.)) < 0.001);
    assert!((as_slice[[1, 0]] - 3_f32.powf(3.)) < 0.001);
    assert!((as_slice[[1, 1]] - 4_f32.powf(4.)) < 0.001);
    assert!((as_slice[[2, 0]] - 5_f32.powf(5.)) < 0.001);
    assert!((as_slice[[2, 1]] - 6_f32.powf(6.)) < 0.001);
}
