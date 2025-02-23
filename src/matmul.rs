use wgpu::{PipelineCompilationOptions, util::DeviceExt};

use crate::{Device, Tensor, query::PerformanceQueries};

pub struct MatMul;

impl Default for MatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl MatMul {
    pub const fn new() -> Self {
        Self
    }
}

impl MatMul {
    fn compile(&self, device: &Device) -> wgpu::ShaderModule {
        let source = template("f32");
        device.create_shader_module(source)
    }

    pub async fn run(
        &self,
        device: &Device,
        a: &Tensor<2, f32>,
        b: &Tensor<2, f32>,
    ) -> Tensor<2, f32> {
        self.run_with_query(device, a, b, None).await
    }

    pub async fn run_with_query(
        &self,
        device: &Device,
        a: &Tensor<2, f32>,
        b: &Tensor<2, f32>,
        query: Option<&PerformanceQueries>,
    ) -> Tensor<2, f32> {
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        let output_buf = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (a_shape[0] * b_shape[1] * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_tensor = Tensor::new_from_buffer(device, output_buf, [a_shape[0], b_shape[1]]);
        self.run_with_query_and_out_tensor(device, a, b, query, &output_tensor)
            .await;
        output_tensor
    }

    pub async fn run_with_query_and_out_tensor(
        &self,
        device: &Device,
        a: &Tensor<2, f32>,
        b: &Tensor<2, f32>,
        query: Option<&PerformanceQueries>,
        output_tensor: &Tensor<2, f32>,
    ) {
        assert_eq!(a.layout().shape()[1], b.layout().shape()[0]);
        let module = self.compile(device);

        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        assert_eq!(*output_tensor.layout().shape(), [a_shape[0], b_shape[1]]);

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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    module: &module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: PipelineCompilationOptions::default(),
                });

        let dims = device
            .wgpu_device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[
                    a_shape[0] as u32,
                    a_shape[1] as _,
                    b_shape[1] as _,
                ]),
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
                        resource: dims.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_tensor.buffer().as_entire_binding(),
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
            const WORKGROUP_SIZE: u32 = 16;
            const TILE_SIZE: u32 = 8;
            let workgroup_size_a = (a_shape[0] as u32).div_ceil(TILE_SIZE * WORKGROUP_SIZE);
            let workgroup_size_b = (b_shape[1] as u32).div_ceil(TILE_SIZE * WORKGROUP_SIZE);
            cpass.dispatch_workgroups(workgroup_size_a, workgroup_size_b, 1);
        }
        if let Some(query) = query {
            query.resolve(&mut encoder);
        }
        device.wgpu_queue().submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul() {
    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let query = PerformanceQueries::new(&device);
    let tensor = MatMul
        .run_with_query(&device, &tensor_a, &tensor_b, Some(&query))
        .await;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);
    println!("{}", query.wait_for_results().await);

    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_matmul() {
    use rand::Rng;

    let device = Device::new().await.unwrap();
    std::thread::spawn({
        let device = device.clone();
        move || loop {
            device.wgpu_device().poll(wgpu::Maintain::Wait);
        }
    });
    let max_size = if cfg!(debug_assertions) { 5 } else { 125 };
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let size1 = rand::rng().random_range(1..max_size);
        let size2 = rand::rng().random_range(1..max_size);
        let size3 = rand::rng().random_range(1..max_size);

        let data_a: Vec<Vec<f32>> = (0..size1)
            .map(|_| (0..size2).map(|_| rand::random()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..size2)
            .map(|_| (0..size3).map(|_| rand::random()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let mut ndarray_a = ndarray::Array2::zeros((size1, size2));
        for i in 0..size1 {
            for j in 0..size2 {
                ndarray_a[[i, j]] = data_a[i][j];
            }
        }
        let mut ndarray_b = ndarray::Array2::zeros((size2, size3));
        for i in 0..size2 {
            for j in 0..size3 {
                ndarray_b[[i, j]] = data_b[i][j];
            }
        }

        let dot = ndarray_a.dot(&ndarray_b);

        let tensor = MatMul.run(&device, &tensor_a, &tensor_b).await;
        let as_slice = tensor.as_slice().await.unwrap();
        for i in 0..size1 {
            for j in 0..size3 {
                assert_eq!(as_slice[[i, j]], dot[[i, j]]);
            }
        }
    }
    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = MatMul.run(&device, &tensor_a, &tensor_b).await;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{:?}", as_slice);

    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 6.);
}

fn template(dtype: &str) -> String {
    format!(
        r#"// From https://github.com/zanussbaum/surfgrad/blob/4f696e3ddcbeb2c7686bc7dbb83c3dbb89595591/src/shaders/matmul.ts
// Article https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel

struct Dimensions {{
  M: u32,
  K: u32,
  N: u32,
}}

@group(0) @binding(0) var<uniform> dimensions: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<{dtype}>;
@group(0) @binding(2) var<storage, read> b: array<{dtype}>;
@group(0) @binding(3) var<storage, read_write> result: array<{dtype}>;

const BLOCKSIZE: u32 = 16u;
const TILE_M: u32 = 8u;  // Tile size in M dimension
const TILE_N: u32 = 8u;  // Tile size in N dimension

@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let row = global_id.y * TILE_M;
    let col = global_id.x * TILE_N;

    var sums: array<array<{dtype}, TILE_N>, TILE_M>;

    // Compute the 2D tile
    for (var k = 0u; k < dimensions.K; k++) {{
      let a_00 = a[row * dimensions.K + k];
      let a01 = a[(row + 1u) * dimensions.K + k];
      let a02 = a[(row + 2u) * dimensions.K + k];
      let a03 = a[(row + 3u) * dimensions.K + k];
      let a04 = a[(row + 4u) * dimensions.K + k];
      let a05 = a[(row + 5u) * dimensions.K + k];
      let a06 = a[(row + 6u) * dimensions.K + k];
      let a07 = a[(row + 7u) * dimensions.K + k];
      let b_00 = b[k * dimensions.N + col];
      let b01 = b[k * dimensions.N + (col + 1u)];
      let b02 = b[k * dimensions.N + (col + 2u)];
      let b03 = b[k * dimensions.N + (col + 3u)];
      let b04 = b[k * dimensions.N + (col + 4u)];
      let b05 = b[k * dimensions.N + (col + 5u)];
      let b06 = b[k * dimensions.N + (col + 6u)];
      let b07 = b[k * dimensions.N + (col + 7u)];
      sums[0][0] += a_00 * b_00;
      sums[0][1] += a_00 * b01;
      sums[0][2] += a_00 * b02;
      sums[0][3] += a_00 * b03;
      sums[0][4] += a_00 * b04;
      sums[0][5] += a_00 * b05;
      sums[0][6] += a_00 * b06;
      sums[0][7] += a_00 * b07;
      sums[1][0] += a01 * b_00;
      sums[1][1] += a01 * b01;
      sums[1][2] += a01 * b02;
      sums[1][3] += a01 * b03;
      sums[1][4] += a01 * b04;
      sums[1][5] += a01 * b05;
      sums[1][6] += a01 * b06;
      sums[1][7] += a01 * b07;
      sums[2][0] += a02 * b_00;
      sums[2][1] += a02 * b01;
      sums[2][2] += a02 * b02;
      sums[2][3] += a02 * b03;
      sums[2][4] += a02 * b04;
      sums[2][5] += a02 * b05;
      sums[2][6] += a02 * b06;
      sums[2][7] += a02 * b07;
      sums[3][0] += a03 * b_00;
      sums[3][1] += a03 * b01;
      sums[3][2] += a03 * b02;
      sums[3][3] += a03 * b03;
      sums[3][4] += a03 * b04;
      sums[3][5] += a03 * b05;
      sums[3][6] += a03 * b06;
      sums[3][7] += a03 * b07;
      sums[4][0] += a04 * b_00;
      sums[4][1] += a04 * b01;
      sums[4][2] += a04 * b02;
      sums[4][3] += a04 * b03;
      sums[4][4] += a04 * b04;
      sums[4][5] += a04 * b05;
      sums[4][6] += a04 * b06;
      sums[4][7] += a04 * b07;
      sums[5][0] += a05 * b_00;
      sums[5][1] += a05 * b01;
      sums[5][2] += a05 * b02;
      sums[5][3] += a05 * b03;
      sums[5][4] += a05 * b04;
      sums[5][5] += a05 * b05;
      sums[5][6] += a05 * b06;
      sums[5][7] += a05 * b07;
      sums[6][0] += a06 * b_00;
      sums[6][1] += a06 * b01;
      sums[6][2] += a06 * b02;
      sums[6][3] += a06 * b03;
      sums[6][4] += a06 * b04;
      sums[6][5] += a06 * b05;
      sums[6][6] += a06 * b06;
      sums[6][7] += a06 * b07;
      sums[7][0] += a07 * b_00;
      sums[7][1] += a07 * b01;
      sums[7][2] += a07 * b02;
      sums[7][3] += a07 * b03;
      sums[7][4] += a07 * b04;
      sums[7][5] += a07 * b05;
      sums[7][6] += a07 * b06;
      sums[7][7] += a07 * b07;
    }}

    // Row 0
    if (row < dimensions.M) {{
        if (col < dimensions.N) {{
            result[row * dimensions.N + col] = sums[0][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[row * dimensions.N + (col + 1u)] = sums[0][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[row * dimensions.N + (col + 2u)] = sums[0][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[row * dimensions.N + (col + 3u)] = sums[0][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[row * dimensions.N + (col + 4u)] = sums[0][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[row * dimensions.N + (col + 5u)] = sums[0][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[row * dimensions.N + (col + 6u)] = sums[0][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[row * dimensions.N + (col + 7u)] = sums[0][7];
        }}
    }}

    // Row 1
    if (row + 1u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 1u) * dimensions.N + col] = sums[1][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 1u)] = sums[1][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 2u)] = sums[1][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 3u)] = sums[1][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 4u)] = sums[1][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 5u)] = sums[1][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 6u)] = sums[1][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 1u) * dimensions.N + (col + 7u)] = sums[1][7];
        }}
    }}

    // Row 2
    if (row + 2u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 2u) * dimensions.N + col] = sums[2][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 1u)] = sums[2][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 2u)] = sums[2][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 3u)] = sums[2][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 4u)] = sums[2][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 5u)] = sums[2][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 6u)] = sums[2][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 2u) * dimensions.N + (col + 7u)] = sums[2][7];
        }}
    }}

    // Row 3
    if (row + 3u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 3u) * dimensions.N + col] = sums[3][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 1u)] = sums[3][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 2u)] = sums[3][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 3u)] = sums[3][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 4u)] = sums[3][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 5u)] = sums[3][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 6u)] = sums[3][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 3u) * dimensions.N + (col + 7u)] = sums[3][7];
        }}
    }}
    if (row + 4u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 4u) * dimensions.N + col] = sums[4][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 1u)] = sums[4][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 2u)] = sums[4][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 3u)] = sums[4][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 4u)] = sums[4][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 5u)] = sums[4][5];
        }}
        if (col + 6u < dimensions.N) {{ 
            result[(row + 4u) * dimensions.N + (col + 6u)] = sums[4][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 4u) * dimensions.N + (col + 7u)] = sums[4][7];
        }}
    }}
    if (row + 5u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 5u) * dimensions.N + col] = sums[5][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 1u)] = sums[5][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 2u)] = sums[5][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 3u)] = sums[5][3]; 
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 4u)] = sums[5][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 5u)] = sums[5][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 6u)] = sums[5][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 5u) * dimensions.N + (col + 7u)] = sums[5][7];
        }}
    }}
    if (row + 6u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 6u) * dimensions.N + col] = sums[6][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 1u)] = sums[6][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 2u)] = sums[6][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 3u)] = sums[6][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 4u)] = sums[6][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 5u)] = sums[6][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 6u)] = sums[6][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 6u) * dimensions.N + (col + 7u)] = sums[6][7];
        }}
    }}
    if (row + 7u < dimensions.M) {{
        if (col < dimensions.N) {{
            result[(row + 7u) * dimensions.N + col] = sums[7][0];
        }}
        if (col + 1u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 1u)] = sums[7][1];
        }}
        if (col + 2u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 2u)] = sums[7][2];
        }}
        if (col + 3u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 3u)] = sums[7][3];
        }}
        if (col + 4u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 4u)] = sums[7][4];
        }}
        if (col + 5u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 5u)] = sums[7][5];
        }}
        if (col + 6u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 6u)] = sums[7][6];
        }}
        if (col + 7u < dimensions.N) {{
            result[(row + 7u) * dimensions.N + (col + 7u)] = sums[7][7];
        }}
    }}
}}"#
    )
}
