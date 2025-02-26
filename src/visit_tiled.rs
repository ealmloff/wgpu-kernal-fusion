use std::fmt::Write;

use wgpu::CommandEncoder;

use crate::{
    DataTypeEnum, PerformanceQueries, TensorData,
    kernel::{GenericKernel, TensorInput},
};

pub(crate) struct VisitTiledKernel<const T: usize> {
    rank: u32,
    contiguous: bool,
    tile_size: u32,
    kernel: GenericKernel,
}

impl<const T: usize> VisitTiledKernel<T> {
    pub(crate) fn new(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatype: DataTypeEnum,
        modify_data: impl FnMut(&mut GenericKernel, [String; T], &[TensorInput; T]) -> String,
    ) -> Self {
        const { assert!(T > 0) }
        let mut kernel = GenericKernel::new();
        let kernel_text = Self::build_tiled_map_kernel(
            rank,
            tile_size,
            contiguous,
            datatype,
            &mut kernel,
            modify_data,
        );
        kernel.set_body(kernel_text);
        let blocksize = Self::blocksize_raw(contiguous, rank);
        let workgroup_size = if contiguous {
            [blocksize, 1, 1]
        } else {
            std::array::from_fn(|i| if T > i { blocksize } else { 1 })
        };
        kernel.set_workgroup_size(workgroup_size);
        Self {
            rank,
            contiguous,
            kernel,
            tile_size,
        }
    }

    fn blocksize_raw(contiguous: bool, rank: u32) -> u32 {
        if contiguous {
            256
        } else {
            // max_blocksize^R = 256
            (256f64.powf(1. / rank as f64)).floor() as u32
        }
    }

    fn blocksize(&self) -> u32 {
        Self::blocksize_raw(self.contiguous, self.rank)
    }

    fn build_tiled_map_kernel(
        rank: u32,
        tile_size: u32,
        contiguous: bool,
        datatype: DataTypeEnum,
        mut kernel: &mut GenericKernel,
        mut modify_data: impl FnMut(&mut GenericKernel, [String; T], &[TensorInput; T]) -> String,
    ) -> String {
        assert!(rank <= 3, "TensorLayout only supports up to 3 rank tensors");

        let mut kernel_body = String::new();
        let global_id = kernel.global_id();
        let tensors = std::array::from_fn(|_| kernel.add_tensor_input(rank, true, datatype));

        if contiguous {
            for local_index in 0..tile_size {
                let index = format!("index_{local_index}");
                writeln!(
                    &mut kernel_body,
                    "\t\tlet {index} = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                write!(&mut kernel_body, "\t\tif {index} < ").unwrap();
                for i in 0..rank {
                    let shape = tensors[0].shape_binding(i);
                    write!(&mut kernel_body, "{shape}").unwrap();
                    if i < rank - 1 {
                        write!(&mut kernel_body, " * ").unwrap();
                    }
                }
                writeln!(&mut kernel_body, " {{").unwrap();
                let modify_data = modify_data(
                    &mut kernel,
                    std::array::from_fn(|_| index.clone()),
                    &tensors,
                );
                writeln!(&mut kernel_body, "{modify_data}").unwrap();
                writeln!(&mut kernel_body, "}}").unwrap();
            }
        } else {
            for i in 0..rank as usize {
                let index = ["x", "y", "z"][i];
                writeln!(
                    &mut kernel_body,
                    "\tlet tile_index_{i} = {global_id}.{index} * {tile_size};"
                )
                .unwrap();
            }
            writeln!(&mut kernel_body, "\n").unwrap();

            for i in 0..rank {
                writeln!(&mut kernel_body, "for (var local_index_{i} = 0u; local_index_{i} < {tile_size}; local_index_{i}++) {{").unwrap();
            }

            for i in 0..rank {
                writeln!(
                    &mut kernel_body,
                    "let merged_index_{i} = tile_index_{i} + local_index_{i};"
                )
                .unwrap();
            }

            write!(&mut kernel_body, "if ").unwrap();
            for i in 0..rank {
                let shape = tensors[0].shape_binding(i);
                write!(&mut kernel_body, "merged_index_{i} < {shape} && ").unwrap();
            }
            writeln!(&mut kernel_body, "true {{").unwrap();
            for (index, tensor) in tensors.iter().enumerate() {
                let offset = tensor.offset_binding();
                write!(&mut kernel_body, "let index_{index} = {offset} + ").unwrap();
                for i in 0..rank {
                    let stride = tensor.stride_binding(i);
                    write!(&mut kernel_body, "{stride} * merged_index_{i} + ").unwrap();
                }
                writeln!(&mut kernel_body, "0;").unwrap();
            }
            let modify_data = modify_data(
                &mut kernel,
                std::array::from_fn(|i| format!("index_{i}")),
                &tensors,
            );
            writeln!(&mut kernel_body, "{modify_data}").unwrap();

            writeln!(&mut kernel_body, "}}").unwrap();

            for _ in 0..rank {
                writeln!(&mut kernel_body, "}}").unwrap();
            }
        }

        kernel_body
    }

    pub(crate) fn run_with_query(
        &self,
        tensors: [&TensorData; T],
        query: Option<&PerformanceQueries>,
        command_encoder: &mut CommandEncoder,
    ) {
        let layout = tensors[0].layout();
        let shape = layout.shape();
        let max_blocksize = self.blocksize();
        let workgroup_dispatch_size = if self.contiguous {
            [
                shape
                    .iter()
                    .map(|x| *x as u32)
                    .product::<u32>()
                    .div_ceil(self.tile_size * max_blocksize),
                1,
                1,
            ]
        } else {
            let workgroup_size_x = shape
                .first()
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_y = shape
                .get(1)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            let workgroup_size_z = shape
                .get(2)
                .map(|x| (*x as u32).div_ceil(self.tile_size * max_blocksize))
                .unwrap_or(1);
            [workgroup_size_x, workgroup_size_y, workgroup_size_z]
        };

        let device = tensors[0].device();
        self.kernel.run_with_query(
            device,
            tensors.iter().map(|x| (*x).clone()),
            query,
            command_encoder,
            workgroup_dispatch_size,
        );
    }
}
