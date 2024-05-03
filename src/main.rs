use crate::{
    config::{DOCS_TO_EMBEDD_FOLDER, DOCS_TO_TRANSLATE_FOLDER, TRANSLATE}, 
    controllers::{keyword_decorator::decorate_passages, translator::translate}, docs::loader::load_data
};
use anyhow::Result;
use config::EMBEDD;

mod llm;
mod config;
mod docs;
mod util;
mod controllers;

fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    
    if TRANSLATE {
        println!("Loading translation docs...");
        let docs = match load_data(DOCS_TO_TRANSLATE_FOLDER) {
            Ok(i) => i,
            Err(e) => panic!("Error loading doc: {:#?}", e),
        };
        translate(docs);
    }
    
    if EMBEDD {
        println!("Loading embedding docs...");
        let docs = match load_data(DOCS_TO_EMBEDD_FOLDER) {
            Ok(i) => i,
            Err(e) => panic!("Error loading docs: {:#?}", e),
        };
        decorate_passages(docs);
    }

}