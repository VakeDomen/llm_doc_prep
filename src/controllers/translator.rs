use std::{cmp::min, sync::{Arc, Mutex}};
use candle_core::Device;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::{
    config::{PAR_CHUNK_SIZE, TRANSLATION_MODEL, TRANSLATION_TOKENIZER, TRANSLATOR_PROGRESS_FILE, TRANSLATOR_SYSTEM_MSG}, 
    docs::{doc::Doc, embedded_doc, saver::{save_raw, save_to_json}}, 
    llm::{model::load_model, prompt::{prompt_model, Prompt}, tokenizer::load_tokenizer}, 
    util::{get_progress_bar, load_progress, save_progress, Progress}
};
use super::splitter::{merge_parsed_documents, split_to_prompts};

pub type ProcessedDocumentChunk = (String, String, bool);

pub fn translate(mut docs: Vec<Doc>) {
    println!("Docs to translate: {}", docs.len());
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

    let tokenizer = match load_tokenizer(TRANSLATION_TOKENIZER) {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };

    let model1 = match load_model(TRANSLATION_MODEL, &device1) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let model2 = match load_model(TRANSLATION_MODEL, &device2) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let mut progress: Progress = load_progress(TRANSLATOR_PROGRESS_FILE);

    
    let to_process = if let Some(bound) = progress.files_to_process {
        min(bound, docs.len())
    } else {
        docs.len()
    };

    let mut done = progress.batches_done * progress.par_chunk_size;
    let progress_bar = get_progress_bar(to_process, 0);
    progress_bar.inc(done); 
    
    docs.drain(0..(done as usize));

    for batch in docs.chunks(progress.par_chunk_size as usize) {

        let results: Vec<(String, Vec<ProcessedDocumentChunk>)> = batch.par_iter().enumerate().map(|(index, document)| {

            // Select the appropriate model and device based on the index
            let (mut model, device) = match index % 2 {
                0 => (model1.lock().unwrap(), &device1),
                _ => (model2.lock().unwrap(), &device2),  
            };

            let mut responses: Vec<ProcessedDocumentChunk> = vec![]; 
            let prompts = split_to_prompts(document);
            let prompts_len = prompts.len();
            let doc_progress = get_progress_bar(prompts_len, 1);
            
            for prompt_string in prompts {
                // Process the prompt with the selected model and device
                let question = prompt_string.clone();
                let prompt = Prompt::One(
                    TRANSLATOR_SYSTEM_MSG.to_string(),
                    prompt_string
                );
                match prompt_model(&mut *model, &tokenizer, prompt, device) {
                    Ok(out) => responses.push((question, out, true)),
                    Err(e) => responses.push((question, e.to_string(), false)),
                };
                doc_progress.inc(1);
            }

            (document.file_name.clone(), responses)

        }).collect();

        progress_bar.inc(PAR_CHUNK_SIZE); 
        done += progress.par_chunk_size;

        progress.batches_done += 1;


        for (file, records) in &results {
            if let Err(e) = save_to_json(&records, &format!("{file}.jsonl")) {
                println!("Failed saving records: {:#?}", e)
            };
        }

        for (file, records) in results {
            let tranlsated_content = merge_parsed_documents(records);
            if let Err(e) = save_raw(tranlsated_content, format!("{file}_translated.md")) {
                println!("Failed saving records: {:#?}", e)
            };
        } 

        if let Err(e) = save_progress(&progress, TRANSLATOR_PROGRESS_FILE) {
            println!("Failed to save progress file: {:#?}", e);
        }

        if done >= to_process as u64 {
            break;
        }
    }

    progress_bar.finish_with_message("Processing complete!");
}

