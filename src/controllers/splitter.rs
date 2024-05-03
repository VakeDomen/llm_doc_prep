use crate::{config::EMBEDDING_TOKENIZER, docs::doc::Doc, llm::tokenizer::load_tokenizer};

use super::translator::ProcessedDocumentChunk;


pub fn split_to_prompts(document: &Doc) -> Vec<String> {
    let tokenizer = match load_tokenizer(EMBEDDING_TOKENIZER) {
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


pub fn split_partial_overlapping(document: &Doc) -> Vec<String> {
    let tokenizer = match load_tokenizer(EMBEDDING_TOKENIZER) {
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

        if current_token_count + token_count > 450 {
            chunks.push(current_chunk.clone());
            current_chunk = if token_count < 100 {
                line.to_string()
            } else {
                String::new()
            };
            current_token_count = 0;
        }
    }

    // Don't forget to add the last chunk if it's not empty
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
    

}




pub fn merge_parsed_documents(records: Vec<ProcessedDocumentChunk>) -> String {
    let mut merged = "".to_owned();
    for (_, translation, success) in records {
        if success {
            merged = format!("{}\n{}", merged, translation);
        }
    }
    merged
}
