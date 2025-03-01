# WGPU kernel fusion

This is a WGPU ML runtime with kernel fusion for ergonomic high performance custom operations. This will hopefully serve as the web and amd runtime for [kalosm](https://crates.io/crates/kalosm) once it is stable enough.

TODO:

- [x] Elementwise ops
- [x] Fuse Elementwise ops together
- [x] MatMul
- [x] Reduce ops
- [x] Fuse Elementwise ops into Reduce ops
- [x] PairWise ops
- [x] Fuse Elementwise ops into PairWise ops
- [ ] Analyze buffer usage for in-place ops
- [x] Memory move/cat/etc ops
- [x] Cast ops
- [ ] Fuse PairWise ops together?
- [ ] Fuse parallel Reduce ops?
- [ ] Dynamically apply fusion based on runtime throughput data

Llama Op Requirements:

- [ ] RmsNorm
- [ ] Matmul
- [ ] Rope
- [ ] Unqueeze
- [x] Cat
- [x] Reshape
- [ ] Transpose
- [ ] Softmax last dim
- [ ] sdpa
- [ ] narraw
- [ ] broadcast add
- [x] silu
- [ ] arange
- [x] sin
- [x] cos
- [ ] rotary_emb::rope
- [ ] rotary_emb::rope_i

## Resources

- https://github.com/googlefonts/compute-shader-101
- https://google.github.io/tour-of-wgsl/
- https://www.w3.org/TR/WGSL/
