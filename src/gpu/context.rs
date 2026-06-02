//! Lazy, process-wide GPU context (device + queue).
//!
//! The context is created on first use and cached for the life of the process.
//! If no compatible adapter exists, every GPU block fails with a clear message
//! rather than panicking.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    /// Compute pipelines keyed by full WGSL source — a shader is compiled once
    /// and reused across ops, blocks, and (crucially) loop iterations.
    pub pipelines: RefCell<HashMap<String, Arc<wgpu::ComputePipeline>>>,
}

static CTX: OnceLock<Option<Mutex<GpuContext>>> = OnceLock::new();

/// Get (initializing on first call) the shared GPU context.
pub fn context() -> Result<&'static Mutex<GpuContext>, String> {
    CTX.get_or_init(GpuContext::try_new)
        .as_ref()
        .ok_or_else(|| "GPU block requires a compatible GPU; no adapter found on this system.".to_string())
}

impl GpuContext {
    fn try_new() -> Option<Mutex<GpuContext>> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("mathlang-gpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .ok()?;
            Some(Mutex::new(GpuContext { device, queue, pipelines: RefCell::new(HashMap::new()) }))
        })
    }
}
