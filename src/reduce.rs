use std::{cell::OnceCell, fmt::Display};

use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use crate::{
    layout::{TensorLayout, TILE_SIZE},
    tensor::DataType,
    Tensor,
};

#[derive(Clone)]
pub(crate) struct ReduceTensorLayout<const R: usize> {
    pub(crate) data: Box<[u32]>,
}

impl<const R: usize> ReduceTensorLayout<R> {
    pub(crate) fn for_tensors<D: DataType>(
        input_tensor: &Tensor<R, D>,
        output_tensor: &Tensor<R, D>,
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
        for i in 0..R {
            kernel.push_str(&format!("\tin_stride_{}: u32,\n", i));
        }
        kernel.push_str(&format!("\tin_offset: u32,\n"));
        for i in 0..R {
            kernel.push_str(&format!("\tout_stride_{}: u32,\n", i));
        }
        kernel.push_str(&format!("\tout_offset: u32,\n"));
        for i in 0..R {
            kernel.push_str(&format!("\tout_shape_{}: u32,\n", i));
        }
        kernel.push_str("}\n");
    }
}

pub struct ReduceOperation {
    dtype: String,
    reduce: ReduceFunction,
    dense_kernel: OnceCell<wgpu::ShaderModule>,
    sparse_kernel: OnceCell<wgpu::ShaderModule>,
}

impl ReduceOperation {
    pub fn new(reduce: ReduceFunction) -> Self {
        Self {
            dtype: "f32".to_string(),
            reduce,
            dense_kernel: OnceCell::new(),
            sparse_kernel: OnceCell::new(),
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

    fn tiled_map<const R: usize>(&self, blocksize: u32, inline: bool) -> String {
        let dtype = &self.dtype;
        let mut kernel = String::new();
        TensorLayout::<R>::wgsl_type_definition(&mut kernel);
        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // And the mlx implementation https://github.com/ml-explore/mlx/blob/b05bcfd27f5f1293401b74dce02e38c8fd7ef66a/mlx/backend/metal/kernels/arg_reduce.metal
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        // We also can't synchronize among workgroups without atomics. storageBarrier() is a barrier for
        // the storage memory only inside the workgroup.
        // This kernel just uses one workgroup per reduction unit like the MLX kernel
        kernel.push_str(&format!("const BLOCKSIZE: u32 = {blocksize}u;\n"));
        kernel.push_str(&format!("const TILE_SIZE: u32 = {TILE_SIZE}u;\n"));
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
        kernel.push_str("fn main(@builtin(workgroup_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_id: u32, @builtin(subgroup_size) subgroup_size: u32, @builtin(subgroup_invocation_id) subgroup_local_id: u32) {\n");
        kernel.push_str("\tconst thread_index = global_id.x;\n");
        kernel.push_str("\tconst workgroup_local_index = thread_index % BLOCKSIZE;\n");
        kernel.push_str("\tconst subgroup_id = workgroup_local_index / subgroup_size;\n");
        kernel.push_str("\t\tconst subgroups_per_workgroup = BLOCKSIZE / subgroup_size;\n");

        // Each thread group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
        kernel.push_str("\tconst thread_group_index = global_id.x / BLOCKSIZE;\n");
        kernel.push_str("\tvar thread_group_index_remainder = thread_group_index;\n");
        kernel.push_str("\tvar in_start_offset = 0;\n");
        kernel.push_str("\tvar out_start_offset = 0;\n");
        for i in 0..R {
            kernel.push_str(&format!(
                "\tconst index_{i} = thread_group_index_remainder % tensor_layout.shape_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tthread_group_index_remainder /= tensor_layout.shape_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tin_start_offset += tensor_layout.in_stride_{i} * index_{i};\n"
            ));
            kernel.push_str(&format!(
                "\tout_start_offset += tensor_layout.out_stride_{i} * index_{i};\n"
            ));
        }
        kernel.push_str("\n");

        kernel.push_str(&format!("\tval merged = {};\n", self.reduce.initial_value));

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        kernel.push_str("\tconst bucket_size = reduce_axis_size / BLOCKSIZE + u32((reduce_axis_size % BLOCKSIZE) == 0);\n");
        // Then loop over this thread's portion of the column and merge the values
        kernel.push_str("\tfor (var index = 0; index < bucket_size; index += 1) {\n");
        kernel.push_str("\t\tconst axis_index = local_id * bucket_size + index;\n");
        kernel
            .push_str("\t\tconst in_index = in_start_offset + axis_index * reduce_axis_stride;\n");
        kernel.push_str("\t\tconst b = input_tensor[in_index];\n");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t}\n");
        kernel.push_str("\n");

        // Next merge within each subgroup with shuffle down
        kernel.push_str("\tfor (var offset = subgroup_size / 2; offset > 0; offset /= 2) {\n");
        kernel.push_str("\t\tconst neighbor = shuffle_down(merged, offset);\n");
        kernel.push_str("\t\tconst b = neighbor;\n");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t}\n");
        kernel.push_str("\n");

        // Write the output to the workgroup memory if this is the first thread in the subgroup
        kernel.push_str("\tif subgroup_local_id == 0 {\n");
        kernel.push_str("\t\tlocal_data[subgroup_id] = merged;\n");
        kernel.push_str("\t}\n");

        // Wait until all threads have written to the workgroup shared memory
        kernel.push_str("workgroupBarrier();\n");

        // Then if this is the first subgroup, do one final shuffle down reduction
        kernel.push_str("\tif subgroup_id == 0 {\n");
        // Copy over the best value from each subgroup from the workgroup shared memory to the merged variable
        kernel.push_str("\t\tif subgroup_local_id < subgroups_per_workgroup {\n");
        kernel.push_str("\t\t\tmerged = local_data[subgroup_local_id];\n");
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t\tfor (var offset = subgroup_size / 2; offset > 0; offset /= 2) {\n");
        kernel.push_str("\t\t\tconst neighbor = shuffle_down(merged, offset);\n");
        kernel.push_str("\t\t\tconst b = neighbor;\n");
        self.modify_element(inline, &mut kernel);
        kernel.push_str("\t\t}\n");

        // Write the output to the output tensor if this is the first thread in the workgroup
        kernel.push_str("\t\tif workgroup_local_index == 0 {\n");
        kernel.push_str("\t\t\toutput_tensor[global_id.x] = merged;\n");
        kernel.push_str("\t\t}\n");
        kernel.push_str("\t}\n");

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
                let source = self.tiled_map::<R>(max_blocksize, true);
                tensor.device().create_shader_module(source)
            })
        } else {
            self.sparse_kernel.get_or_init(|| {
                let source = self.tiled_map::<R>(max_blocksize, true);
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
