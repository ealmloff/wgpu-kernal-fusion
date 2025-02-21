use std::{cell::OnceCell, fmt::Display};

use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use crate::{tensor::DataType, Tensor};

#[derive(Clone)]
pub(crate) struct ReduceTensorLayout<const R1: usize, const R2: usize> {
    pub(crate) data: Box<[u32]>,
}

impl<const R1: usize, const R2: usize> ReduceTensorLayout<R1, R2> {
    pub(crate) fn for_tensors<D: DataType>(
        input_tensor: &Tensor<R1, D>,
        output_tensor: &Tensor<R2, D>,
    ) -> Self {
        let input_layout = *input_tensor.layout();
        let output_layout = *output_tensor.layout();
        let data = input_layout
            .strides()
            .iter()
            .map(|x| *x as u32)
            .chain(std::iter::once(input_layout.offset() as u32))
            .chain(output_layout.strides().iter().map(|x| *x as u32))
            .chain(std::iter::once(output_layout.offset() as u32))
            .chain(output_layout.shape().iter().map(|x| *x as u32))
            .collect();

        Self { data }
    }

    pub(crate) fn wgsl_type_definition(kernel: &mut String) {
        kernel.push_str("struct ReduceTensorLayout {\n");
        for i in 0..R1 {
            kernel.push_str(&format!("\tin_stride_{}: u32,\n", i));
        }
        kernel.push_str(&format!("\tin_offset: u32,\n"));
        for i in 0..R2 {
            kernel.push_str(&format!("\tout_stride_{}: u32,\n", i));
        }
        kernel.push_str(&format!("\tout_offset: u32,\n"));
        for i in 0..R2 {
            kernel.push_str(&format!("\tout_shape_{}: u32,\n", i));
        }
        kernel.push_str("}\n");
    }
}

pub struct ReduceOperation {
    dtype: String,
    reduce: ReduceFunction,
    kernel: OnceCell<wgpu::ShaderModule>,
}

impl ReduceOperation {
    pub fn new(reduce: ReduceFunction) -> Self {
        Self {
            dtype: "f32".to_string(),
            reduce,
            kernel: OnceCell::new(),
        }
    }

    fn dtype(&self) -> String {
        self.dtype.clone()
    }

    fn modify_element(&self, inline: bool, kernel: &mut String) {
        if !inline {
            let call = self.reduce.call("merged", "b");

            kernel.push_str(&format!("merged = {call};\n"));
        } else {
            kernel.push_str(&self.reduce.operation);
            kernel.push('\n');
        }
    }

    fn tiled_map<const R1: usize, const R2: usize>(&self, blocksize: u32, inline: bool) -> String {
        let dtype = &self.dtype;
        let mut kernel = String::new();
        ReduceTensorLayout::<R1, R2>::wgsl_type_definition(&mut kernel);
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
            kernel.push_str(&self.reduce.function(&self.dtype));
        }
        kernel.push_str("\n@compute @workgroup_size(BLOCKSIZE)\n");
        kernel.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_id: u32, @builtin(subgroup_size) subgroup_size: u32) {\n");
        kernel.push_str("\tlet thread_index = global_id.x;\n");
        kernel.push_str("\tlet workgroup_local_index = thread_index % BLOCKSIZE;\n");
        kernel.push_str("\tlet subgroup_id = workgroup_local_index / subgroup_size;\n");
        kernel.push_str("\tlet subgroup_local_id = workgroup_local_index % subgroup_size;\n");
        kernel.push_str("\tlet subgroups_per_workgroup = BLOCKSIZE / subgroup_size;\n");

        // Each workgroup group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
        kernel.push_str("\tlet workgroup_index = global_id.x / BLOCKSIZE;\n");
        kernel.push_str("\tvar workgroup_index_remainder = workgroup_index;\n");
        kernel.push_str("\tvar in_start_offset = 0u;\n");
        kernel.push_str("\tvar out_start_offset = 0u;\n");
        for i in 0..R2 {
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
        kernel.push_str("\n");

        kernel.push_str(&format!("\tvar merged = {};\n", self.reduce.initial_value));

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        kernel.push_str("\tlet bucket_size = reduce_axis_size / BLOCKSIZE + u32((reduce_axis_size % BLOCKSIZE) != 0u);\n");
        // Then loop over this thread's portion of the column and merge the values
        kernel.push_str("\tfor (var index = 0u; index < bucket_size; index += 1u) {\n");
        kernel.push_str("\t\tlet axis_index = workgroup_local_index * bucket_size + index;\n");
        kernel.push_str("\t\tif axis_index < reduce_axis_size {\n");
        kernel.push_str("\t\t\tlet in_index = in_start_offset + axis_index * reduce_axis_stride;\n");
        kernel.push_str("\t\t\tlet b = input_tensor[in_index];\n");
        kernel.push_str("\t\t\t");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t}\n");
        kernel.push_str("\n");

        // Next merge within each subgroup with shuffle down
        kernel.push_str("\tfor (var offset = subgroup_size / 2u; offset > 0u; offset /= 2u) {\n");
        kernel.push_str("\t\tlet neighbor = subgroupShuffleDown(merged, offset);\n");
        kernel.push_str("\t\tlet b = neighbor;\n");
        kernel.push_str("\t\t");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t}\n");
        kernel.push_str("\n");

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
        kernel.push_str(&format!("\t\t\tmerged = {};\n", self.reduce.initial_value));
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t\tfor (var offset = subgroup_size / 2u; offset > 0u; offset /= 2u) {\n");
        kernel.push_str("\t\t\tlet neighbor = subgroupShuffleDown(merged, offset);\n");
        kernel.push_str("\t\t\tlet b = neighbor;\n");
        kernel.push_str("\t\t\t");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t\t}\n");

        // Write the output to the output tensor if this is the first thread in the workgroup
        kernel.push_str("\t\tif workgroup_local_index == 0u {\n");
        kernel.push_str("\t\t\toutput_tensor[global_id.x] = merged;\n");
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t}\n");

        kernel.push_str("}\n");

        println!("{}", kernel);

        kernel
    }

    pub fn run(&self, tensor: &Tensor<2, f32>, dim: usize) -> Tensor<1, f32> {
        // let max_blocksize = tensor.layout().shape()[dim].min(256) as u32;
        let max_blocksize = 256;
        let module = self.kernel.get_or_init(|| {
            let source = self.tiled_map::<2, 1>(max_blocksize, true);
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
        let new_tensor_shape =
            std::array::from_fn(|i| if i < dim { shape[i] } else { shape[i + 1] });
        let output_buf = tensor
            .device()
            .wgpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (new_tensor_shape.iter().product::<usize>() * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        let output_tensor = Tensor::new_from_buffer(tensor.device(), output_buf, new_tensor_shape);

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

        let layout = ReduceTensorLayout::for_tensors(tensor, &output_tensor);

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
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = output_tensor.layout().shape().iter().product::<usize>() as u32;
            cpass.dispatch_workgroups(workgroup_size, 1, 1)
        }
        tensor.device().wgpu_queue().submit(Some(encoder.finish()));

        output_tensor
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

    fn function(&self, dtype: &str) -> String {
        let Self {
            name_id, operation, ..
        } = self;
        format!(
            r#"fn reduce_{name_id}(a: {dtype}, b: {dtype}) -> {dtype} {{
    var merged = a;
    {operation}
    return merged;
}}"#
        )
    }
}

fn add() -> ReduceFunction {
    ReduceFunction::new("merged = merged + b;".to_string(), "0.0")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_add() {
    use crate::Device;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let add = add();
    let reduction = ReduceOperation::new(add);
    let output = reduction.run(&tensor, 1);

    let output = output.as_slice().await.unwrap();
    println!("{:?}", output);
    assert_eq!(output[[0]], 3.);
    assert_eq!(output[[1]], 7.);
    assert_eq!(output[[2]], 11.);
}
