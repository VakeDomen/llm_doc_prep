use std::{fs::{File, OpenOptions}, io::Write, path::Path};
use anyhow::Result;
use csv::Writer;

use crate::{util::get_progress_bar, ProcessedDocumentChunk};


pub fn save_to_csv(records: Vec<ProcessedDocumentChunk>, file_name: &str) -> Result<()> {
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

pub fn save_to_json(records: &Vec<ProcessedDocumentChunk>, file_name: &str) -> Result<()> {
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
    Ok(())
}

pub fn save_raw(content: String, file_name: String) -> Result<()> {
    let mut f = File::create(&file_name)?;
    f.write_all(content.as_bytes())?;
    Ok(())
}