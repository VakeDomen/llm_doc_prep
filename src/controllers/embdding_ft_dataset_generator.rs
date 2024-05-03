use std::{cmp::min, sync::{Arc, Mutex}};
use candle_core::Device;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tokio::runtime::Runtime;
use crate::{
    config::{KEYWORD_DECORATOR_MODEL, KEYWORD_DECORATOR_PROGRESS_FILE, KEYWORD_DECORATOR_SYSTEM_MSG, KEYWORD_DECORATOR_TOKENIZER, PAR_CHUNK_SIZE}, 
    docs::{doc::Doc, embedded_doc::{EmbeddedDoc, Passage}, qdant::insert_docs, saver::{save_raw, save_to_json}}, 
    llm::{embedding_model::embedd, model::load_model, prompt::{prompt_model, Prompt}, tokenizer::load_tokenizer}, 
    util::{get_progress_bar, load_progress, save_progress, Progress}
};
use super::splitter::split_partial_overlapping;

pub type ProcessedDocumentChunk = (String, String, bool);

pub fn decorate_passages(mut passages: Vec<Doc>) {
    println!("Passages to decorate: {}", passages.len());
    let device1 = match Device::new_cuda(0) {
        Ok(cuda) => cuda,
        Err(e) => {
            println!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
            Device::Cpu
        },
    };

    let device2 = match Device::new_cuda(1) {
        Ok(cuda) => cuda,
        Err(e) => {
            println!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
            Device::Cpu
        },
    };

    let tokenizer = match load_tokenizer(KEYWORD_DECORATOR_TOKENIZER) {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };

    let model1 = match load_model(KEYWORD_DECORATOR_MODEL, &device1) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let model2 = match load_model(KEYWORD_DECORATOR_MODEL, &device2) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let mut progress: Progress = load_progress(KEYWORD_DECORATOR_PROGRESS_FILE);

    
    let to_process = if let Some(bound) = progress.files_to_process {
        min(bound, passages.len())
    } else {
        passages.len()
    };

    let mut done = progress.batches_done * progress.par_chunk_size;
    let progress_bar = get_progress_bar(to_process, 0);
    progress_bar.inc(done); 
    
    passages.drain(0..(done as usize));

    for batch in passages.chunks(progress.par_chunk_size as usize) {

        let decorated_docs: Vec<Vec<EmbeddedDoc>> = batch.par_iter().enumerate().map(|(index, document)| {

            // Select the appropriate model and device based on the index
            let (mut model, device) = match index % 2 {
                0 => (model1.lock().unwrap(), &device1),
                _ => (model2.lock().unwrap(), &device2),  
            };

            let mut responses: Vec<ProcessedDocumentChunk> = vec![]; 
            let prompts = split_partial_overlapping(document);
            let prompts_len = prompts.len();
            let doc_progress = get_progress_bar(prompts_len, 1);
            
            for prompt_string in prompts {
                // Process the prompt with the selected model and device
                let question = prompt_string.clone();
                let prompt = Prompt::One(
                    KEYWORD_DECORATOR_SYSTEM_MSG.to_string(),
                    format!(
                        "Name of the file: {}\nPassage: {}\n\n Response template: 'KW: <kw1>, <kw2>, <kw3>,...'", 
                        document.file_name, 
                        prompt_string
                    )
                );
                match prompt_model(&mut *model, &tokenizer, prompt, device) {
                    Ok(out) => responses.push((question, out, true)),
                    Err(e) => responses.push((question, e.to_string(), false)),
                };
                doc_progress.inc(1);
            }

            let mut embedded_docs = vec![];

            let rt = Runtime::new().unwrap();  // Create a new Tokio runtime
            for (passage, keywords, success) in responses {
                if success {
                    let content = format!("{}\n\n{}", keywords, passage);
                    let embedding_vector = match rt.block_on(async { embedd(&content).await }) {
                        Ok(vec) => vec.to_vec2::<f32>(),
                        Err(e) => {
                            println!("Ccant embedd passage: {:#?}\n{}",e, passage);
                            continue;
                        },
                    };

                    let embedding_vector = match embedding_vector {
                        Ok(v) => v,
                        Err(e) =>  {
                            println!("Cant convert passage embedding tensor: {:#?}\n{}",e, passage);
                            continue;
                        },
                    };

                    let vector = embedding_vector.get(0).unwrap().clone();

                    embedded_docs.push(EmbeddedDoc {
                        vector,
                        content: Passage {
                            usage: 0,
                            text: content,
                        }
                    });
                }
            }

            match rt.block_on(async { insert_docs(embedded_docs.clone()).await }) {
                Ok(_) => (),
                Err(e) => {
                    println!("Error upserting to Qdrant: {:#?}", e);
                    ()
                },
            }

            embedded_docs
        }).collect();

        

        progress_bar.inc(PAR_CHUNK_SIZE); 
        done += progress.par_chunk_size;

        progress.batches_done += 1;


        if let Err(e) = save_progress(&progress, KEYWORD_DECORATOR_PROGRESS_FILE) {
            println!("Failed to save progress file: {:#?}", e);
        }

        if done >= to_process as u64 {
            break;
        }
    }

    progress_bar.finish_with_message("Deorating passages complete!");
}

