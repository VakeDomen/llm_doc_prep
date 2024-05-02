use crate::{config::{DOCS_TO_TRANSLATE_FOLDER, TRANSLATE}, controllers::translator::translate, docs::loader::load_data};


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
        println!("Loading doc");
        let docs = match load_data(DOCS_TO_TRANSLATE_FOLDER) {
            Ok(i) => i,
            Err(e) => panic!("Error loading doc: {:#?}", e),
        };
        translate(docs);
    }
    
}