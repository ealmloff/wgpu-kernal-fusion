use std::{fmt::Display, marker::PhantomData, sync::OnceLock};

use wgpu::{PipelineCompilationOptions, util::DeviceExt};

use crate::{
    Tensor, UntypedElementWiseOperation,
    layout::Layout,
    query::PerformanceQueries,
    tensor::{DataType, DataTypeEnum, TensorData, padded_tensor_size},
};

#[derive(Clone)]
pub(crate) struct ReduceTensorLayout {
    pub(crate) data: Box<[u32]>,
}

impl ReduceTensorLayout {
    pub(crate) fn new(dim: usize, input_layout: &Layout, output_layout: &Layout) -> Self {
        let data = input_layout
            .strides()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x as u32))
            .chain(std::iter::once(input_layout.offset() as u32))
            .chain(output_layout.strides().iter().map(|x| *x as u32))
            .chain(std::iter::once(output_layout.offset() as u32))
            .chain(output_layout.shape().iter().map(|x| *x as u32))
            .collect();

        Self { data }
    }

    pub(crate) fn output_rank(&self) -> usize {
        (self.data.len() - 1) / 3
    }

    pub(crate) fn wgsl_type_definition(&self, kernel: &mut String) {
        let rank = self.output_rank();
        kernel.push_str("struct ReduceTensorLayout {\n");
        for i in 0..rank {
            kernel.push_str(&format!("\tin_stride_{}: u32,\n", i));
        }
        kernel.push_str("\tin_offset: u32,\n");
        for i in 0..rank {
            kernel.push_str(&format!("\tout_stride_{}: u32,\n", i));
        }
        kernel.push_str("\tout_offset: u32,\n");
        for i in 0..rank {
            kernel.push_str(&format!("\tout_shape_{}: u32,\n", i));
        }
        kernel.push_str("}\n");
    }
}

pub struct ReduceOperation<T> {
    untyped: UntypedReduceOperation,
    datatype: PhantomData<T>,
}

impl<T: DataType> ReduceOperation<T> {
    pub fn new(reduce: ReduceFunction) -> Self {
        Self {
            untyped: UntypedReduceOperation::new(reduce, T::WGSL_TYPE),
            datatype: PhantomData,
        }
    }

    pub fn run(&self, tensor: &Tensor<2, T>, dim: usize) -> Tensor<1, T> {
        self.run_with_query(tensor, dim, None)
    }

    pub fn run_with_query(
        &self,
        tensor: &Tensor<2, T>,
        dim: usize,
        query: Option<&PerformanceQueries>,
    ) -> Tensor<1, T> {
        self.untyped
            .run_with_query(tensor.data(), dim, query)
            .into()
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        tensor: &Tensor<2, T>,
        dim: usize,
        query: Option<&PerformanceQueries>,
        output_tensor: &Tensor<1, T>,
    ) {
        self.untyped
            .run_with_query_and_out_tensor(tensor.data(), dim, query, output_tensor.data())
    }
}

pub(crate) struct UntypedReduceOperation {
    pre_element_wise: UntypedElementWiseOperation,
    reduce: ReduceFunction,
    post_element_wise: UntypedElementWiseOperation,
    kernel: OnceLock<wgpu::ShaderModule>,
    datatype: DataTypeEnum,
}

impl UntypedReduceOperation {
    pub fn new(reduce: ReduceFunction, datatype: DataTypeEnum) -> Self {
        Self {
            pre_element_wise: UntypedElementWiseOperation::empty(datatype),
            reduce,
            post_element_wise: UntypedElementWiseOperation::empty(datatype),
            kernel: OnceLock::new(),
            datatype,
        }
    }

    fn merge(&self, inline: bool, kernel: &mut String) {
        if !inline {
            let call = self.reduce.call("merged", "data");

            kernel.push_str(&format!("merged = {call};\n"));
        } else {
            kernel.push_str(&self.reduce.operation);
            kernel.push('\n');
        }
    }

    fn tiled_map(&self, blocksize: u32, inline: bool, layout: &ReduceTensorLayout) -> String {
        let dtype = self.datatype;
        let mut kernel = String::new();
        if dtype == DataTypeEnum::F16 {
            kernel.push_str("enable f16;\n");
        }
        layout.wgsl_type_definition(&mut kernel);
        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // And the mlx implementation https://github.com/ml-explore/mlx/blob/b05bcfd27f5f1293401b74dce02e38c8fd7ef66a/mlx/backend/metal/kernels/arg_reduce.metal
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        // We also can't synchronize among workgroups without atomics. storageBarrier() is a barrier for
        // the storage memory only inside the workgroup.
        // This kernel just uses one workgroup per reduction unit like the MLX kernel
        kernel.push_str(&format!("const BLOCKSIZE: u32 = {blocksize}u;\n"));
        kernel.push_str("@group(0) @binding(0) var<uniform> tensor_layout: ReduceTensorLayout;\n");
        kernel.push_str("@group(0) @binding(1) var<uniform> reduce_axis_stride: u32;\n");
        kernel.push_str("@group(0) @binding(2) var<uniform> reduce_axis_size: u32;\n");
        kernel.push_str(&format!(
            "@group(0) @binding(3) var<storage, read> input_tensor: array<{dtype}>;\n"
        ));
        kernel.push_str(&format!(
            "@group(0) @binding(4) var<storage, read_write> output_tensor: array<{dtype}>;\n"
        ));
        kernel.push_str(&format!(
            "var<workgroup> local_data: array<{dtype}, BLOCKSIZE>;\n"
        ));
        if !inline {
            kernel.push_str(&self.reduce.function(dtype));
        }
        self.pre_element_wise.add_functions(inline, &mut kernel);
        self.post_element_wise.add_functions(inline, &mut kernel);
        kernel.push_str("\n@compute @workgroup_size(BLOCKSIZE)\n");
        kernel.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(subgroup_size) subgroup_size: u32) {\n");
        kernel.push_str("\tlet thread_index = global_id.x;\n");
        kernel.push_str("\tlet workgroup_local_index = thread_index % BLOCKSIZE;\n");
        kernel.push_str("\tlet subgroup_id = workgroup_local_index / subgroup_size;\n");
        kernel.push_str("\tlet subgroup_local_id = workgroup_local_index % subgroup_size;\n");
        kernel.push_str("\tlet subgroups_per_workgroup = BLOCKSIZE / subgroup_size;\n");

        // Each workgroup group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
        kernel.push_str("\tlet workgroup_index = global_id.x / BLOCKSIZE;\n");
        kernel.push_str("\tvar workgroup_index_remainder = workgroup_index;\n");
        kernel.push_str("\tvar in_start_offset = tensor_layout.in_offset;\n");
        kernel.push_str("\tvar out_start_offset = tensor_layout.out_offset;\n");
        for i in 0..layout.output_rank() {
            kernel.push_str(&format!(
                "\tlet index_{i} = workgroup_index_remainder % tensor_layout.out_shape_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tworkgroup_index_remainder /= tensor_layout.out_shape_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tin_start_offset += tensor_layout.in_stride_{i} * index_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tout_start_offset += tensor_layout.out_stride_{i} * index_{i};\n"
            ));
        }
        kernel.push('\n');

        kernel.push_str(&format!(
            "\tvar merged = {dtype}({});\n",
            self.reduce.initial_value
        ));

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        kernel.push_str("\tlet bucket_size = reduce_axis_size / BLOCKSIZE + u32((reduce_axis_size % BLOCKSIZE) != 0u);\n");
        // Then loop over this thread's portion of the column and merge the values
        kernel.push_str("\tfor (var index = 0u; index < bucket_size; index += 1u) {\n");
        kernel.push_str("\t\tlet axis_index = workgroup_local_index * bucket_size + index;\n");
        kernel.push_str("\t\tif axis_index < reduce_axis_size {\n");
        kernel
            .push_str("\t\t\tlet in_index = in_start_offset + axis_index * reduce_axis_stride;\n");
        kernel.push_str("\t\t\tvar data = input_tensor[in_index];\n");
        self.pre_element_wise.modify_data(inline, &mut kernel);
        kernel.push_str("\t\t\t");
        self.merge(inline, &mut kernel);
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t}\n");
        kernel.push('\n');

        // Next merge within each subgroup with shuffle down
        kernel.push_str("\tfor (var offset = subgroup_size / 2u; offset > 0u; offset /= 2u) {\n");
        kernel.push_str("\t\tlet neighbor = subgroupShuffleDown(merged, offset);\n");
        kernel.push_str("\t\tlet data = neighbor;\n");
        kernel.push_str("\t\t");
        self.merge(inline, &mut kernel);
        kernel.push_str("\t}\n");
        kernel.push('\n');

        // Write the output to the workgroup memory if this is the first thread in the subgroup
        kernel.push_str("\tif subgroup_local_id == 0u {\n");
        kernel.push_str("\t\tlocal_data[subgroup_id] = merged;\n");
        kernel.push_str("\t}\n");

        // Wait until all threads have written to the workgroup shared memory
        kernel.push_str("\tworkgroupBarrier();\n");

        // Then if this is the first subgroup, do one final shuffle down reduction
        kernel.push_str("\tif subgroup_id == 0u {\n");
        // Copy over the best value from each subgroup from the workgroup shared memory to the merged variable
        kernel.push_str("\t\tif subgroup_local_id < subgroups_per_workgroup {\n");
        kernel.push_str("\t\t\tmerged = local_data[subgroup_local_id];\n");
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t\telse {\n");
        kernel.push_str(&format!(
            "\t\t\tmerged = {dtype}({});\n",
            self.reduce.initial_value
        ));
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t\tfor (var offset = subgroup_size / 2u; offset > 0u; offset /= 2u) {\n");
        kernel.push_str("\t\t\tlet neighbor = subgroupShuffleDown(merged, offset);\n");
        kernel.push_str("\t\t\tlet data = neighbor;\n");
        kernel.push_str("\t\t\t");
        self.merge(inline, &mut kernel);
        kernel.push_str("\t\t}\n");

        // Write the output to the output tensor if this is the first thread in the workgroup
        kernel.push_str("\t\tif workgroup_local_index == 0u {\n");
        kernel.push_str("\t\t\tvar data = merged;\n");
        self.post_element_wise.modify_data(inline, &mut kernel);
        kernel.push_str("\t\t\toutput_tensor[out_start_offset] = data;\n");
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t}\n");

        kernel.push_str("}\n");

        kernel
    }

    pub fn run_with_query(
        &self,
        tensor: &TensorData,
        dim: usize,
        query: Option<&PerformanceQueries>,
    ) -> TensorData {
        let shape = tensor.layout().shape();
        let new_tensor_shape = shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let output_buf = tensor
            .device()
            .wgpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: padded_tensor_size(
                    (new_tensor_shape.iter().product::<usize>() * self.datatype.element_size())
                        as u64,
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        let output_tensor = TensorData::new_from_buffer(
            tensor.device(),
            output_buf,
            &new_tensor_shape,
            self.datatype,
        );

        self.run_with_query_and_out_tensor(tensor, dim, query, &output_tensor);

        output_tensor
    }

    pub fn run_with_query_and_out_tensor(
        &self,
        tensor: &TensorData,
        dim: usize,
        query: Option<&PerformanceQueries>,
        output_tensor: &TensorData,
    ) {
        assert_eq!(
            *output_tensor.layout().shape(),
            [tensor
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| { (i != dim).then_some(*x as u32) })
                .product::<u32>() as usize]
        );

        let limits = tensor.device().wgpu_device().limits();
        let max_blocksize = (tensor.layout().shape()[dim] as u32)
            .min(limits.max_compute_workgroup_size_x)
            .max(limits.min_subgroup_size)
            .max(32);
        let layout = ReduceTensorLayout::new(dim, tensor.layout(), output_tensor.layout());
        let module = self.kernel.get_or_init(|| {
            let source = self.tiled_map(max_blocksize, true, &layout);
            tensor.device().create_shader_module(source)
        });

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
                            ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
        let shape = tensor.layout().shape();
        let strides = tensor.layout().strides();

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
                module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
            },
        );

        let layout =
            tensor
                .device()
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&layout.data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let axis_stride_binding =
            tensor
                .device()
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&strides[dim]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let axis_size_binding =
            tensor
                .device()
                .wgpu_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&shape[dim]),
                    usage: wgpu::BufferUsages::UNIFORM,
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
                            resource: axis_stride_binding.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: axis_size_binding.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: tensor.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output_tensor.buffer().as_entire_binding(),
                        },
                    ],
                });

        let mut encoder = tensor
            .device()
            .wgpu_device()
            .create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: query.map(|query| query.compute_timestamp_writes()),
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = output_tensor.layout().shape().iter().product::<usize>() as u32;
            cpass.dispatch_workgroups(workgroup_size, 1, 1)
        }
        if let Some(query) = query {
            query.resolve(&mut encoder);
        }
        tensor.device().wgpu_queue().submit(Some(encoder.finish()));
    }
}

pub struct ReduceFunction {
    name_id: u64,
    operation: String,
    initial_value: String,
}

impl ReduceFunction {
    fn new(operation: impl Display, initial_value: impl Display) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            name_id,
            operation: operation.to_string(),
            initial_value: initial_value.to_string(),
        }
    }

    fn call(&self, a: impl Display, b: impl Display) -> String {
        let name_id = self.name_id;
        format!("reduce_{name_id}({a}, {b})")
    }

    fn function(&self, dtype: DataTypeEnum) -> String {
        let Self {
            name_id, operation, ..
        } = self;
        format!(
            r#"fn reduce_{name_id}(a: {dtype}, data: {dtype}) -> {dtype} {{
    var merged = a;
    {operation}
    return merged;
}}"#
        )
    }
}

pub fn sum() -> ReduceFunction {
    ReduceFunction::new("merged = merged + data;".to_string(), "0.0")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let add = sum();
    let reduction = ReduceOperation::new(add);
    let query = PerformanceQueries::new(&device);
    let output = reduction.run_with_query(&tensor, 0, Some(&query));

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    println!("{}", query.wait_for_results().await);
    assert_eq!(output[[0]], 9.);
    assert_eq!(output[[1]], 12.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 3.);
    assert_eq!(output[[1]], 7.);
    assert_eq!(output[[2]], 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum_f16() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor = Tensor::new(&device, &data);

    let add = sum();
    let reduction = ReduceOperation::new(add);
    let query = PerformanceQueries::new(&device);
    let output = reduction.run_with_query(&tensor, 0, Some(&query));

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    println!("{}", query.wait_for_results().await);
    assert_eq!(output[[0]], half::f16::from_f32(9.));
    assert_eq!(output[[1]], half::f16::from_f32(12.));

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], half::f16::from_f32(3.));
    assert_eq!(output[[1]], half::f16::from_f32(7.));
    assert_eq!(output[[2]], half::f16::from_f32(11.));
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sliced_sum() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.slice([0..3, 0..1]);

    let add = sum();
    let reduction = ReduceOperation::new(add);
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 9.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_add_then_sum_fused() {
    use crate::{Device, ElementWiseOperation, add_const};

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let add = sum();
    let mut reduction = ReduceOperation::new(add);
    reduction.untyped.pre_element_wise = ElementWiseOperation::<f32>::new([add_const(1.0)]).untyped;
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 3. + 9.);
    assert_eq!(output[[1]], 3. + 12.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2. + 3.);
    assert_eq!(output[[1]], 2. + 7.);
    assert_eq!(output[[2]], 2. + 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_sum_then_add_fused() {
    use crate::{Device, ElementWiseOperation, add_const};

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let add = sum();
    let mut reduction = ReduceOperation::new(add);
    reduction.untyped.post_element_wise =
        ElementWiseOperation::<f32>::new([add_const(1.0)]).untyped;
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1. + 9.);
    assert_eq!(output[[1]], 1. + 12.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1. + 3.);
    assert_eq!(output[[1]], 1. + 7.);
    assert_eq!(output[[2]], 1. + 11.);
}

pub fn max() -> ReduceFunction {
    ReduceFunction::new("merged = max(merged, data);".to_string(), "-3.40282e+38")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_max() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let max = max();
    let reduction = ReduceOperation::new(max);
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 5.);
    assert_eq!(output[[1]], 6.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 4.);
    assert_eq!(output[[2]], 6.);
}

pub fn min() -> ReduceFunction {
    ReduceFunction::new("merged = min(merged, data);".to_string(), "3.40282e+38")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_min() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let min = min();
    let reduction = ReduceOperation::new(min);
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 2.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

pub fn product() -> ReduceFunction {
    ReduceFunction::new("merged = merged * data;".to_string(), "1.0")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_product() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let product = product();
    let reduction = ReduceOperation::new(product);
    let output = reduction.run(&tensor, 0);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 15.);
    assert_eq!(output[[1]], 48.);

    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 12.);
    assert_eq!(output[[2]], 30.);
}
