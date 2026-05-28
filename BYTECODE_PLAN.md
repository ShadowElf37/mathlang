The key insight connecting both systems: Vec<Instruction> is not just a CPU speedup —
  it's the natural IR for GPU lambda lowering. The bytecode VM and the GPU backend were
  always the same feature from opposite ends.

  Nested lambdas plan (MakeClosure instruction):
  - Compile-time: walk the inner body for free variables from the outer params/locals;
  push their values onto the stack; compile the inner body with placeholder captured
  entries (so it emits LoadCaptured not inlined constants); emit
  MakeClosure(inner_params, body, pre_compiled_code, free_var_names)
  - Runtime: pop the free-var values, build new_captured = outer_captured + {name: val},
   create a Val::Fn with the pre-filled OnceLock — zero recompile cost on first call
  - Needs one new constructor Val::make_fn_compiled() to pre-fill the OnceLock

  GPU interface (section 15 in CONSIDERATIONS.md):
  - GPU-safe subset of Instructions maps cleanly to WGSL (table in the doc)
  - gpu/vm_lower.rs converts a GPU-safe Vec<Instruction> to an inline WGSL function body
  - tensor((i,j)->expr, m, n) on GPU: compile lambda → check is_gpu_safe() → lower to
  WGSL → emit GpuOp::TensorFromLambda → dispatch m*n threads
  - Falls back to CPU if lambda isn't GPU-safe — safe default, GPU lowering is always an
   optimization
  - T[i,j] inside GPU lambdas deferred to v3 (requires a new LoadBuffer instruction)
