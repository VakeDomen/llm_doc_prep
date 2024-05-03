use crate::{
    config::{DOCS_TO_TRANSLATE_FOLDER, TRANSLATE}, 
    controllers::translator::translate, docs::loader::load_data, llm::embedding_model::generate_prompt_embedding
};
use anyhow::Result;

mod llm;
mod config;
mod docs;
mod util;
mod controllers;

#[tokio::main]
async fn main() -> Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    
    if TRANSLATE {
        println!("Loading doc");
        let docs = match load_data(DOCS_TO_TRANSLATE_FOLDER) {
            Ok(i) => i,
            Err(e) => panic!("Error loading doc: {:#?}", e),
        };
        translate(docs);
    }
    
    

    match generate_prompt_embedding("Hello world").await {
        Ok(em) => println!("{:?} -> {:?}",em.shape(), em.to_vec2::<f32>()),
        Err(e) => println!("ERROR: {:#?}", e)
    };
    Ok(())
}