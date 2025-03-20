use anyhow::{Error as E, Result};
use candle_transformers::models::stable_diffusion;
use candle_core::{Device, DType, Tensor};
use tokenizers::Tokenizer;
use candle_core::Module;

use crate::ai::model_files::ModelFile;
use crate::ai::text_encoder;
use crate::configs::TEXT_ENCODER_WEIGHT;


pub fn build_tokenizer(tokenizer_path: &str, padding: &Option<String>) 
    -> Result<(Tokenizer, u32)> {
    let tokenizer_file = ModelFile::Tokenizer.get(tokenizer_path.to_string())?;
    let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let pad_id = match padding {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    Ok((tokenizer, pad_id))
}

pub fn generate_text_embeddings(
    prompt: &str,
    neg_prompt: &str,
    tokenizer: &Tokenizer,
    tokenizer_pad_id: u32,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(tokenizer_pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let text_model = text_encoder::build_text_encoder(
        TEXT_ENCODER_WEIGHT, &sd_config.clip, device)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut neg_tokens = tokenizer
            .encode(neg_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if neg_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                neg_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while neg_tokens.len() < sd_config.clip.max_position_embeddings {
            neg_tokens.push(tokenizer_pad_id)
        }

        let neg_tokens = Tensor::new(neg_tokens.as_slice(), device)?.unsqueeze(0)?;
        let neg_embeddings = text_model.forward(&neg_tokens)?;

        Tensor::cat(&[neg_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}
