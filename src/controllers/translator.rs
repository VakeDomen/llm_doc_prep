use std::{cmp::min, sync::{Arc, Mutex}};

use candle_core::Device;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{config::{PAR_CHUNK_SIZE, TRANSLATOR_PROGRESS_FILE}, docs::{doc::Doc, saver::{save_raw, save_to_json}}, llm::{model::load_model, prompt::{prompt_model, Prompt}, tokenizer::load_tokenizer}, util::{get_progress_bar, load_progress, save_progress, Progress}};

pub type ProcessedDocumentChunk = (String, String, bool);


pub fn translate(mut docs: Vec<Doc>) {
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

    let tokenizer = match load_tokenizer("models/llama3-8b/tokenizer.json") {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };

    let model1 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device1) { 
        Ok(m) => Arc::new(Mutex::new(m)),
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let model2 = match load_model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device2) { 
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
    let progress_bar = get_progress_bar(to_process);
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
            let doc_progress = get_progress_bar(prompts_len);
            
            for prompt_string in prompts {
                // Process the prompt with the selected model and device
                let question = prompt_string.clone();
                let prompt = Prompt::One(prompt_string);
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


fn split_to_prompts(document: &Doc) -> Vec<String> {
    let tokenizer = match load_tokenizer("models/llama3-8b/tokenizer.json") {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {}", e),
    };

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    // Split the document content by newlines
    let lines = document.content.split("\n\n");

    for line in lines {
        // Use tokenizer to encode the line and check the token count
        let tokens = match tokenizer.encode(&*line, true) {
            Ok(t) => t,
            Err(e) => panic!("Error tokenizing: {}", e),
        };
        let token_count = tokens.len();

        // Add this line to the current chunk
        if !current_chunk.is_empty() {
            current_chunk.push('\n');
            current_chunk.push('\n');
        }
        current_chunk.push_str(&line);
        current_token_count += token_count;

        // Check if adding this line exeeded the token limit
        if current_token_count + token_count > 450 {
            // If current chunk is full, push it to chunks and start a new one
            chunks.push(current_chunk.clone());
            current_chunk = String::new();
            current_token_count = 0;
        }
    }

    // Don't forget to add the last chunk if it's not empty
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
    

}



fn merge_parsed_documents(records: Vec<ProcessedDocumentChunk>) -> String {
    let mut merged = "".to_owned();
    for (_, translation, success) in records {
        if success {
            merged = format!("{}\n{}", merged, translation);
        }
    }
    merged
}
