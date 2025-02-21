# WGPU kernel fusion

This is a WGPU ML runtime with kernel fusion for ergonomic high performance custom operations. This will hopefully serve as the web and amd runtime for [kalosm](https://crates.io/crates/kalosm) once it is stable enough.

TODO:
- [x] Elementwise ops
- [x] Fuse Elementwise ops together
- [x] MatMul
- [ ] Reduce ops
- [ ] Fuse Elementwise ops into Reduce ops
- [ ] Dynamically apply fusion based on runtime throughput data
