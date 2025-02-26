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
- [ ] Memory move/cat/etc ops
- [ ] Cast ops
- [ ] Fuse PairWise ops together?
- [ ] Fuse parallel Reduce ops?
- [ ] Dynamically apply fusion based on runtime throughput data

## Resources

- https://github.com/googlefonts/compute-shader-101
- https://google.github.io/tour-of-wgsl/
- https://www.w3.org/TR/WGSL/
