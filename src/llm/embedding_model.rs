use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::rotary_emb::{self, rope};

use crate::llm::loader::load_bert_model;


/// Generates a normalized L2 embedding for a given text prompt using a pre-loaded BERT model.
///
/// This function loads a BERT model and tokenizer, encodes the input prompt into token IDs, and uses the model
/// to generate word embeddings. The embeddings are then pooled (averaged) and normalized to produce a single
/// vector representation of the prompt, suitable for further processing or similarity comparisons.
///
/// # Parameters
/// - `prompt`: The text prompt to be embedded.
///
/// # Returns
/// Returns a `Result` containing the normalized tensor representing the L2 embedding of the prompt.
///
/// # Errors
/// - Returns an error if any step in the embedding generation process fails, including model loading, tokenization,
///   tensor operations, or WebSocket communication failures.
pub async fn embedd(
    prompt: &str,
) -> Result<Tensor> {
    let (model, tokenizer, device) = load_bert_model(Some(0))?;
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    
    let embeddings = normalize_l2(&embeddings)?;
    
    
    // ROTATY POSITION EMBEDINGS ATTEMPT
    // // Compute Rotary Embeddings based on the sequence length and embedding dimension.
    // let seq_len = tokens.len();
    // let d_model = 1024;  // Assuming embeddings shape is [1, 1, seq_len, d_model]
    // let (cos, sin) = generate_rotary_embeddings_for_sequence(
    //     seq_len, 
    //     d_model / 2, 
    //     &device
    // )?;
    // let embeddings = embeddings.unsqueeze(0)?;
    // let rotary_embedding = rope(&embeddings, &cos, &sin)?;
    // let rotary_embedding = rotary_embedding.squeeze(0)?;
    // // let rotary_embedding = (rotary_embedding.sum(1)? / (d_model as f64))?;
    // let pooled_embedding = rotary_embedding.mean(1)?;  // Reduces across the sequence length, resulting in [1, d_model]
    // let normalized_embedding = normalize_l2(&pooled_embedding)?;
    Ok(embeddings)
}

fn generate_rotary_embeddings_for_sequence(seq_len: usize, half_d_model: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let freqs: Vec<f32> = (0..half_d_model)
        .map(|i| 10000_f32.powf(-(i as f32) / half_d_model as f32))
        .collect();

    let mut cos_matrix = Vec::with_capacity(seq_len);
    let mut sin_matrix = Vec::with_capacity(seq_len);

    for position in 0..seq_len {
        let angles: Vec<f32> = freqs.iter()
            .map(|&freq| freq * position as f32)
            .collect();

        let cos: Vec<f32> = angles.iter().map(|&angle| angle.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|&angle| angle.sin()).collect();

        cos_matrix.push(Tensor::from_iter(cos.into_iter(), device)?);
        sin_matrix.push(Tensor::from_iter(sin.into_iter(), device)?);
    }

    let cos_tensor = Tensor::stack(&cos_matrix, 0)?;
    let sin_tensor = Tensor::stack(&sin_matrix, 0)?;

    Ok((cos_tensor, sin_tensor))
}

/// Normalizes a tensor using L2 norm.
///
/// This function takes a tensor and normalizes its values across a specified dimension using the L2 norm.
/// This is typically used to normalize embeddings so that they can be more effectively compared or processed
/// downstream.
///
/// # Parameters
/// - `v`: The tensor to be normalized.
///
/// # Returns
/// Returns a `Result` containing the normalized tensor.
///
/// # Errors
/// - Returns an error if tensor operations required for normalization fail.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

