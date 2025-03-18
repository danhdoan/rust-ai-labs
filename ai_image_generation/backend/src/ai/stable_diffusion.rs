use candle_transformers::models::stable_diffusion::{
    self,
    unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};
use candle_core::{DType, D, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;
use tracing::info;

use rand::Rng;
use crate::ai::tokenizer::{build_tokenizer, generate_text_embeddings};
use crate::ai::vae::build_vae_model;
use crate::ai::unet::build_unet_model;
use crate::errors::ErrorCode;
use crate::image_lib;


use crate::configs::{TOKENIZER_PATH, VAE_WEIGHT, UNET_WEIGHT};

pub struct StableDiffusion {
    sd_config: stable_diffusion::StableDiffusionConfig,
    tokenizer: Tokenizer,
    tokenizer_pad_id: u32, 
    vae: AutoEncoderKL,
    unet: UNet2DConditionModel,
    device: Device,
}

impl StableDiffusion {
    pub fn new(device: Device) -> Result<Self, ErrorCode> {
        let seed = rand::rng().random_range(0u64..u64::MAX);
        let _ = device.set_seed(seed)
            .map_err(|_| ErrorCode::Inference);

        // build Stable Diffusion config
        let sliced_attention_size: Option<usize> = None;
        let (height, width) = (768, 768);
        let sd_config = stable_diffusion::StableDiffusionConfig::v2_1(
            sliced_attention_size, Some(height), Some(width)
        );

        // build Stable Diffusion tokenizer
        let (tokenizer, pad_id) = build_tokenizer(
            TOKENIZER_PATH, &sd_config.clip.pad_with)
            .map_err(|_| ErrorCode::Inference)?;

        // build Stable Diffusion VAE
        let vae = build_vae_model(VAE_WEIGHT, &sd_config, &device)
            .map_err(|_| ErrorCode::Inference)?;

        // build Stable Diffusion UNet
        let unet = build_unet_model(UNET_WEIGHT, &sd_config, &device)
            .map_err(|_| ErrorCode::Inference)?;

        Ok(Self {
            sd_config,
            tokenizer,
            tokenizer_pad_id: pad_id,
            vae,
            unet,
            device,
        })
    }

    pub fn run(&self, prompt: &str, uncond_prompt: &str) -> Result<String, ErrorCode> {
        let guidance_scale = 9.0; 
        let use_guide_scale = guidance_scale > 1.0;
        let dtype = DType::F16;

        let n_steps = 20;
        let mut scheduler = self.sd_config.build_scheduler(n_steps)
            .map_err(|_| ErrorCode::Inference)?;

        let bsize = 1;

        let text_embeddings: Vec<Tensor> = vec![
            generate_text_embeddings(
                prompt,
                uncond_prompt,
                &self.tokenizer,
                self.tokenizer_pad_id,
                &self.sd_config,
                &self.device,
                dtype,
                use_guide_scale)
            .map_err(|_| ErrorCode::TextEmbeddingGeneration)?
        ];

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)
            .map_err(|_| ErrorCode::Inference)?;

        let text_embeddings = text_embeddings.repeat((bsize, 1, 1))
            .map_err(|_| ErrorCode::Inference)?;

        let vae_scale = 0.18215;
        let timesteps = scheduler.timesteps().to_vec();
        let latents = Tensor::randn(
            0f32,
            1f32,
            (bsize, 4, self.sd_config.height / 8, self.sd_config.width / 8),
            &self.device)
            .map_err(|_| ErrorCode::Inference)?;
        let latents = (latents * scheduler.init_noise_sigma())
            .map_err(|_| ErrorCode::Inference)?;
        let mut latents = latents.to_dtype(dtype)
            .map_err(|_| ErrorCode::Inference)?;

        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)
                    .expect("Failed to create latent_model_input")
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
                .map_err(|_| ErrorCode::Inference)?;

            let latent_model_input = latent_model_input.to_device(&self.device)
                .map_err(|_| ErrorCode::Inference)?;

            let noise_pred = self.unet.forward(
                &latent_model_input, timestep as f64, &text_embeddings)
                .map_err(|_| ErrorCode::Inference)?;

            let noise_pred = {
                let noise_pred = noise_pred.chunk(2, 0).map_err(|_| ErrorCode::Inference)?;

                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
                let diff = (noise_pred_text - noise_pred_uncond)
                    .map_err(|_| ErrorCode::Inference)?;

                (noise_pred_uncond + diff * guidance_scale)
                    .map_err(|_| ErrorCode::Inference)?
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)
                    .map_err(|_| ErrorCode::Inference)?;

            let dt = start_time.elapsed().as_secs_f32();
            println!("step {} done, {:.2}s", timestep_index + 1, dt);
        }
        info!("Image generation done");


        let image = postprocess(&self.vae, &latents, vae_scale, bsize)
            .map_err(|_| ErrorCode::PostProcessing)?;
        image_lib::image_to_base64(image)
            .map_err(|_| ErrorCode::PostProcessing)
    }
}

fn postprocess(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    _bsize: usize,
) -> anyhow::Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

    // TODO: implement batch processing
    // in this version, only generate 1 image
    let image = images.i(0)?;
    let image = image_lib::tensor_to_image(&image)?;
    Ok(image)
}
