use std::{cell::OnceCell, fmt::Display};

use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use crate::{
    layout::{TensorLayout, TILE_SIZE},
    Tensor,
};

pub struct ReduceOperation {
    dtype: String,
    functions: Vec<ReduceFunction>,
    dense_kernel: OnceCell<wgpu::ShaderModule>,
    sparse_kernel: OnceCell<wgpu::ShaderModule>,
}

impl Default for ReduceOperation {
    fn default() -> Self {
        Self::new([])
    }
}

impl ReduceOperation {
    pub fn new(functions: impl IntoIterator<Item = ReduceFunction>) -> Self {
        Self {
            dtype: "f32".to_string(),
            functions: functions.into_iter().collect(),
            dense_kernel: OnceCell::new(),
            sparse_kernel: OnceCell::new(),
        }
    }

    pub fn then(self, other: ReduceFunction) -> Self {
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

    fn modify_element(&self, inline: bool, kernal: &mut String) {
        if !inline {
            let call = self
                .functions
                .iter()
                .fold("matrix[index]".to_string(), |acc, f| f.call(acc));

            kernal.push_str(&format!("matrix[index] = {call};\n"));
        } else {
            kernal.push_str("var data = matrix[index];\n");
            for function in &self.functions {
                kernal.push_str(&function.operation);
                kernal.push('\n');
            }
            kernal.push_str("matrix[index] = data;\n");
        }
    }

    fn tiled_map<const R: usize>(&self, blocksize: u32, inline: bool) -> String {
        let dtype = &self.dtype;
        let mut kernal = String::new();
        TensorLayout::<R>::wgsl_type_definition(&mut kernal);
        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        kernal.push_str(&format!("const BLOCKSIZE: u32 = {blocksize}u;\n"));
        kernal.push_str(&format!("const TILE_SIZE: u32 = {TILE_SIZE}u;\n"));
        kernal.push_str("@group(0) @binding(0) var<uniform> tensor_layout: TensorLayout;\n");
        kernal.push_str(&format!(
            "@group(0) @binding(1) var<storage, read> input_tensor: array<{dtype}>;\n"
        ));
        kernal.push_str(&format!(
            "@group(0) @binding(2) var<storage, read_write> output_tensor: array<{dtype}>;\n"
        ));
        kernal.push_str(&format!(
            "var<workgroup> workgroup_reduction_data: array<f32, BLOCKSIZE>;\n"
        ));
        if !inline {
            for function in &self.functions {
                kernal.push_str(&function.function(&self.dtype));
            }
        }
        kernal.push_str("\n@compute @workgroup_size(BLOCKSIZE)\n");
        kernal.push_str("fn main(@builtin(workgroup_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(subgroup_size) subgroup_size: u32) {\n");
        kernal.push_str("const thread_index = global_id.x;\n");
        kernal.push_str("const doubled_index = global_id.x;\n");
        kernal.push_str("\n");


        kernal.push_str("}\n");

        kernal
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
}

impl ReduceFunction {
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
        format!("reduce_{name_id}({data})")
    }

    fn function(&self, dtype: &str) -> String {
        let Self { name_id, operation } = self;
        format!(
            r#"fn reduce_{name_id}(a: {dtype}, b: {dtype}) -> {dtype} {{
    var data;
    {operation}
    return data;
}}"#
        )
    }
}
