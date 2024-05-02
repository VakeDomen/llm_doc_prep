
use std::{cmp::min, fs::{File, OpenOptions}, io::Write, panic, path::Path, sync::{Arc, Mutex}, vec};

use candle_core::{self, Device};
use csv::Writer;
use docs::doc::Doc;
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::Result;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
        config::{PAR_CHUNK_SIZE, PROGRESS_FILE}, docs::loader::{load_data, load_jsonl_data}, llm::{
        model::load_model,
        prompt::{prompt_model, Prompt}, 
        tokenizer::load_tokenizer
    }, progress::{load_progress, save_progress, Progress}
};

mod llm;
mod config;
mod docs;
mod progress;

type ProcessedDocumentChunk = (String, String, bool);

fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    println!("Loading doc");
    let mut docs = match load_data("./data/") {
        Ok(i) => i,
        Err(e) => panic!("Error loading doc: {:#?}", e),
    };

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

    let mut progress: Progress = load_progress(PROGRESS_FILE);

    
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

            for prompt_string in split_to_prompts(document) {
                // Process the prompt with the selected model and device
                let question = prompt_string.clone();
                let prompt = Prompt::One(prompt_string);
                match prompt_model(&mut *model, &tokenizer, prompt, device) {
                    Ok(out) => responses.push((question, out, true)),
                    Err(e) => responses.push((question, e.to_string(), false)),
                };
            }

            (document.file_name.clone(), responses)

        }).collect();

        // for result in results {
        //     match result {
        //         Ok((id, out)) => {
        //             processed.push((id, out));
        //         },
        //         Err((id, err)) => failed.push((id, err)),
        //     }
        // }

        progress_bar.inc(PAR_CHUNK_SIZE); 
        done += progress.par_chunk_size;

        progress.batches_done += 1;


        for (file, records) in results {
            if let Err(e) = save_to_json(&records, &format!("./data/{file}.jsonl")) {
                println!("Failed saving records: {:#?}", e)
            };
        }

        if let Err(e) = save_progress(&progress, PROGRESS_FILE) {
            println!("Failed to save progress file: {:#?}", e);
        }

        if done >= to_process as u64 {
            break;
        }
    }

    progress_bar.finish_with_message("Processing complete!");
}

fn split_to_prompts(document: &Doc) -> Vec<String> {
    todo!()
}

fn get_progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64);
    progress_bar.set_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-"));
    progress_bar
}

fn save_to_csv(records: Vec<ProcessedDocumentChunk>, file_name: &str) -> Result<()> {
    let mut wtr = Writer::from_path(file_name)?;
    let mut successful_writes = 0;
    let mut failed_writes = 0;
    println!("Saving file: {}", file_name);
    let progress_bar = get_progress_bar(records.len());
    for record in records.into_iter() {
        match wtr.write_record(&[record.0, record.1]) {
            Ok(_) => successful_writes += 1,
            Err(_) => failed_writes += 1,
        };
        progress_bar.inc(1); 
    }
    wtr.flush()?;
    println!("Saved: {} \tFailed: {}", successful_writes, failed_writes);
    Ok(())
}

fn save_to_json(records: &Vec<ProcessedDocumentChunk>, file_name: &str) -> Result<()> {
    let mut file = if !Path::new(file_name).exists() {
        File::create(file_name)?
    } else {
        OpenOptions::new()
            .write(true)
            .append(true)
            .open(file_name)?
    };

    for record in records {
        if let Err(e) = writeln!(file, "{:?}", serde_json::to_string(record).unwrap()) {
            eprintln!("Couldn't write to file: {}", e);
        }
    }
    // println!("Saving file: {}", file_name);
    

    // // Flushing the buffer to ensure all data is written to the file
    // buf_writer.flush()?;

    // // Update progress bar after serialization
    // progress_bar.finish_with_message("File saved successfully.");

    Ok(())
}
