use candle_transformers::models::stable_diffusion::{
    self,
    unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};
use candle_core::{DType, D, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use anyhow::{Result};
use rand::Rng;
use crate::ai::tokenizer::{build_tokenizer, generate_text_embeddings};
use crate::ai::vae::build_vae_model;
use crate::ai::unet::build_unet_model;
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
    pub fn new(device: Device) -> Result<Self> {
        let seed = rand::rng().random_range(0u64..u64::MAX);
        device.set_seed(seed)?;

        // build Stable Diffusion config
        let sliced_attention_size: Option<usize> = None;
        let (height, width) = (768, 768);
        let sd_config = stable_diffusion::StableDiffusionConfig::v2_1(
            sliced_attention_size, Some(height), Some(width)
        );

        // build Stable Diffusion tokenizer
        let (tokenizer, pad_id) = build_tokenizer(
            TOKENIZER_PATH, &sd_config.clip.pad_with)?;

        // build Stable Diffusion VAE
        let vae = build_vae_model(VAE_WEIGHT, &sd_config, &device)?;

        // build Stable Diffusion UNet
        let unet = build_unet_model(UNET_WEIGHT, &sd_config, &device)?;

        Ok(Self {
            sd_config,
            tokenizer,
            tokenizer_pad_id: pad_id,
            vae,
            unet,
            device,
        })
    }

    pub fn run(&self, prompt: &str, uncond_prompt: &str) -> Result<()> {
        let guidance_scale = 9.0; 
        let use_guide_scale = guidance_scale > 1.0;
        let dtype = DType::F16;

        let n_steps = 30;
        let mut scheduler = self.sd_config.build_scheduler(n_steps)?;

        let bsize = 1;

        let text_embeddings: Vec<Tensor> = vec![generate_text_embeddings(
            &prompt,
            &uncond_prompt,
            &self.tokenizer,
            self.tokenizer_pad_id,
            &self.sd_config,
            &self.device,
            dtype,
            use_guide_scale)?];

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;

        let vae_scale = 0.18215;
        let num_samples = 1;
        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps().to_vec();
            let latents = Tensor::randn(
                0f32,
                1f32,
                (bsize, 4, self.sd_config.height / 8, self.sd_config.width / 8),
                &self.device,
            )?;
            let latents = (latents * scheduler.init_noise_sigma())?;
            let mut latents = latents.to_dtype(dtype)?;

            println!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)
                        .expect("Failed to create latent_model_input")
                } else {
                    latents.clone()
                };

                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
                let latent_model_input = latent_model_input.to_device(&self.device)?;
                let noise_pred = self.unet.forward(
                    &latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {} done, {:.2}s", timestep_index + 1, dt);
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );

            let final_image = "sd_final.png".to_string();
            save_image(
                &self.vae,
                &latents,
                vae_scale,
                bsize,
                idx,
                &final_image,
                num_samples,
                None,
            )?;
        }

        Ok(())
    }
}

fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let image_filename = output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        image_lib::save_image(&image, image_filename)?;
    }
    Ok(())
}
