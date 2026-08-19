# Dialect reference

## hexir

Graph level. Tensors, value semantics.

| Op | Form | Lowers? |
| --- | --- | --- |
| `hexir.constant` | `dense<...> : tensor<...>` | yes |
| `hexir.linear` | `%a, %b : tensor<...>` | yes, to `linalg.matmul` |
| `hexir.add` | `%a, %b : tensor<...>` | yes |
| `hexir.relu` | `%x : tensor<...>` | yes |
| `hexir.print` | `%x : tensor<...>` | yes, to `printf` |
| `hexir.call_tir` | `@callee(%args) : (...) -> ...` | bridge to `hextir` |
| `hexir.func` | function with a body | yes |
| `sigmoid`, `softmax`, `gelu`, `swish`, `mish`, `tanh`, `elu`, `leaky_relu` | | **declared only** |

Ops in the last row parse and verify but have no lowering. Using one gives a
clear error rather than silently wrong code.

### Attributes

`device`
: `"cpu"` or `"cuda"`. Written by the partition pass, copied by every lowering.

### hexir.call_tir

The only way from the graph level to the kernel level.

```mlir
%r = hexir.call_tir @matmul(%a, %b)
     : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
```

Its verifier enforces destination passing: the callee must take one buffer per
argument, plus one for the result.

## hextir

Kernel level. Buffers, explicit loops.

| Op | Purpose |
| --- | --- |
| `hextir.prim_func` | a kernel; last argument is the destination |
| `hextir.return` | terminator of a `prim_func` |
| `hextir.block` | a named, schedulable region |
| `hextir.for` | one loop level, carrying its kind |
| `hextir.yield` | terminator of a `for` or `block` |
| `hextir.alloc_buffer` | a kernel-local buffer |
| `hextir.buffer_load` | read one element |
| `hextir.buffer_store` | write one element |

### hextir.for

```mlir
hextir.for "parallel" %c0 to %cN step %c1 { ... }
hextir.for "thread_binding" %c0 to %cN step %c1 bind "threadIdx.x" { ... }
```

`kind` is one of `serial`, `parallel`, `vectorized`, `unrolled`,
`thread_binding`. This attribute *is* the schedule — a scheduling pass changes
it rather than restructuring the loop nest.

`bind` names the GPU axis, and is only meaningful for `thread_binding`.

### Attributes

`device`
: which device this kernel targets.

`hexir.kernel`
: `"matmul"`, `"add"` or `"relu"`. Says what the kernel computes without
  anyone having to match on its body. The `.hxb` serializer turns this into a
  descriptor.

