enable f16;
@group(0) @binding(0) var<storage, read_write> i_0: array<f32>;
@group(0) @binding(1) var<uniform> i_1: u32;
@group(0) @binding(3) var<uniform> i_3: u32;
@group(0) @binding(2) var<uniform> i_2: u32;
@group(0) @binding(4) var<storage, read_write> i_4: array<f32>;
@group(0) @binding(5) var<uniform> i_5: u32;
@group(0) @binding(7) var<uniform> i_7: u32;
@group(0) @binding(6) var<uniform> i_6: u32;
@group(0) @binding(8) var<storage, read_write> i_8: array<f16>;
@group(0) @binding(9) var<uniform> i_9: u32;
@group(0) @binding(11) var<uniform> i_11: u32;
@group(0) @binding(10) var<uniform> i_10: u32;
fn f_0(input: f32) -> f16 { let output = f16(input); return output; }fn f_1(input: f32) -> f16 { let output = f16(input); return output; }fn f_2(input: f16) -> f16 { let output = -input; return output; }fn f_3(input: f16) -> f16 { let output = exp(input); return output; }fn f_4(input: f16) -> f16 { let output = input + 1; return output; }fn f_5(a: f32, b: f32) -> f16 { let output = a / b; return output; }const BLOCKSIZE: u32 = 256u;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
let index_0 = global_id.x * 8 + 0;
if index_0 < 1 * i_3 {let a = f_0(i_0[index_0]);
let b = f_4(f_3(f_2(f_1(i_4[index_0]))));
i_8[index_0] = f_5(a, b);

}let index_1 = global_id.x * 8 + 1;
if index_1 < 1 * i_3 {let a = f_0(i_0[index_1]);
let b = f_4(f_3(f_2(f_1(i_4[index_1]))));
i_8[index_1] = f_5(a, b);

}let index_2 = global_id.x * 8 + 2;
if index_2 < 1 * i_3 {let a = f_0(i_0[index_2]);
let b = f_4(f_3(f_2(f_1(i_4[index_2]))));
i_8[index_2] = f_5(a, b);

}let index_3 = global_id.x * 8 + 3;
if index_3 < 1 * i_3 {let a = f_0(i_0[index_3]);
let b = f_4(f_3(f_2(f_1(i_4[index_3]))));
i_8[index_3] = f_5(a, b);

}let index_4 = global_id.x * 8 + 4;
if index_4 < 1 * i_3 {let a = f_0(i_0[index_4]);
let b = f_4(f_3(f_2(f_1(i_4[index_4]))));
i_8[index_4] = f_5(a, b);

}let index_5 = global_id.x * 8 + 5;
if index_5 < 1 * i_3 {let a = f_0(i_0[index_5]);
let b = f_4(f_3(f_2(f_1(i_4[index_5]))));
i_8[index_5] = f_5(a, b);

}let index_6 = global_id.x * 8 + 6;
if index_6 < 1 * i_3 {let a = f_0(i_0[index_6]);
let b = f_4(f_3(f_2(f_1(i_4[index_6]))));
i_8[index_6] = f_5(a, b);

}let index_7 = global_id.x * 8 + 7;
if index_7 < 1 * i_3 {let a = f_0(i_0[index_7]);
let b = f_4(f_3(f_2(f_1(i_4[index_7]))));
i_8[index_7] = f_5(a, b);

}
}