use crate::Buffer;
use crate::graphics::gradient;
use crate::quad::{self, Quad};

use bytemuck::{Pod, Zeroable};
use std::ops::Range;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GradientRenderStrategy {
    Linear,
    Radial,
}

#[derive(Debug)]
/// A quad filled with interpolated colors.
pub struct Gradient {
    /// The background gradient data of the quad.
    pub gradient: gradient::Packed,

    /// The [`Quad`] data of the [`Gradient`].
    pub quad: Quad,
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct LinearGradient {
    /// The background gradient data of the quad.
    pub gradient: gradient::LinearPacked,

    /// The [`Quad`] data of the [`Gradient`].
    pub quad: Quad,
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct RadialGradient {
    /// The background gradient data of the quad.
    pub gradient: gradient::RadialPacked,

    /// The [`Quad`] data of the [`Gradient`].
    pub quad: Quad,
}

#[derive(Debug)]
pub struct Layer {
    linear_instances: Buffer<LinearGradient>,
    radial_instances: Buffer<RadialGradient>,
    instance_count: usize,
}

impl Layer {
    pub fn new(device: &wgpu::Device) -> Self {
        let linear_instances = Buffer::new(
            device,
            "iced_wgpu.quad.radial_gradient.buffer",
            quad::INITIAL_INSTANCES,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let radial_instances = Buffer::new(
            device,
            "iced_wgpu.quad.radial_gradient.buffer",
            quad::INITIAL_INSTANCES,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            radial_instances,
            linear_instances,
            instance_count: 0,
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut wgpu::util::StagingBelt,
        instances: &[Gradient],
    ) {
        let linear_instances: Vec<_> = instances
            .iter()
            .filter_map(|gradient| {
                if let gradient::Packed::Linear(linear) = gradient.gradient {
                    return Some(LinearGradient {
                        gradient: linear,
                        quad: gradient.quad,
                    });
                }
                None
            })
            .collect();
        let radial_instances: Vec<_> = instances
            .iter()
            .filter_map(|gradient| {
                if let gradient::Packed::Radial(radial) = gradient.gradient {
                    return Some(RadialGradient {
                        gradient: radial,
                        quad: gradient.quad,
                    });
                }
                None
            })
            .collect();

        if !linear_instances.is_empty() {
            let _ = self.linear_instances.resize(device, linear_instances.len());
            let _ =
                self.linear_instances
                    .write(device, encoder, belt, 0, linear_instances.as_slice());
        }

        if !radial_instances.is_empty() {
            let _ = self.linear_instances.resize(device, linear_instances.len());
            let _ =
                self.radial_instances
                    .write(device, encoder, belt, 0, radial_instances.as_slice());
        }

        self.instance_count = instances.len();
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    #[cfg(not(target_arch = "wasm32"))]
    linear_gradient_pipeline: wgpu::RenderPipeline,
    #[cfg(not(target_arch = "wasm32"))]
    radial_gradient_pipeline: wgpu::RenderPipeline,
}

impl Pipeline {
    #[allow(unused_variables)]
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        constants_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iced_wgpu.quad.gradient.pipeline"),
                push_constant_ranges: &[],
                bind_group_layouts: &[constants_layout],
            });

            // Create linear gradient pipeline
            let linear_gradient_pipeline = Self::create_gradient_pipeline(
                device,
                &layout,
                format,
                "linear",
                include_str!("../shader/quad/gradient_linear.wgsl"),
            );

            // Create radial gradient pipeline
            let radial_gradient_pipeline = Self::create_gradient_pipeline(
                device,
                &layout,
                format,
                "radial",
                include_str!("../shader/quad/gradient_radial.wgsl"),
            );

            Self {
                linear_gradient_pipeline,
                radial_gradient_pipeline,
            }
        }

        #[cfg(target_arch = "wasm32")]
        Self {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn create_gradient_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        gradient_type: &str,
        gradient_shader: &str,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("iced_wgpu.quad.gradient.{}.shader", gradient_type)),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&format!(
                "{}\n{}\n{}\n{}\n{}",
                include_str!("../shader/quad.wgsl"),
                include_str!("../shader/vertex.wgsl"),
                gradient_shader,
                include_str!("../shader/color.wgsl"),
                include_str!("../shader/color/linear_rgb.wgsl")
            ))),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!(
                "iced_wgpu.quad.gradient.{}.pipeline",
                gradient_type
            )),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("gradient_vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Gradient>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array!(
                        // Colors 1-2
                        0 => Uint32x4,
                        // Colors 3-4
                        1 => Uint32x4,
                        // Colors 5-6
                        2 => Uint32x4,
                        // Colors 7-8
                        3 => Uint32x4,
                        // Offsets 1-8
                        4 => Uint32x4,
                        // Direction (for linear) / Center & radii (for radial)
                        5 => Float32x4,
                        // Position & Scale
                        6 => Float32x4,
                        // Border color
                        7 => Float32x4,
                        // Border radius
                        8 => Float32x4,
                        // Border width
                        9 => Float32,
                        // Snap
                        10 => Uint32,
                    ),
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("gradient_fs_main"),
                targets: &quad::color_target_state(format),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }
    #[allow(unused_variables)]
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        constants: &'a wgpu::BindGroup,
        layer: &'a Layer,
        range: Range<usize>,
        strategy: &GradientRenderStrategy,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pipeline = match strategy {
                GradientRenderStrategy::Linear => {
                    render_pass.set_pipeline(&self.linear_gradient_pipeline);
                    render_pass.set_vertex_buffer(0, layer.linear_instances.slice(..));
                }
                GradientRenderStrategy::Radial => {
                    render_pass.set_pipeline(&self.radial_gradient_pipeline);
                    render_pass.set_vertex_buffer(0, layer.radial_instances.slice(..));
                }
            };
            render_pass.set_bind_group(0, constants, &[]);
            render_pass.draw(0..6, range.start as u32..range.end as u32);
        }
    }
}
