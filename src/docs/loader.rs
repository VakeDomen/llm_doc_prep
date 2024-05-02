use std::{fs::{self, File}, io::{BufRead, BufReader, Read}, path::Path};

use anyhow::Result;

use crate::docs::doc::Doc;

pub fn load_jsonl_data(file_name: &str) -> Result<Vec<Doc>> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut docs = vec![];
    for line in reader.lines() {
        let line = line?;
        let doc: Doc = serde_json::from_str(&line)?;
        docs.push(doc)
    }
    Ok(docs)
}


pub fn load_data(folder_name: &str) -> Result<Vec<Doc>> {
    let mut data = Vec::new();
    let path = Path::new(folder_name);

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && (path.extension().map_or(false, |ext| ext == "txt" || ext == "md")) {
            let mut file = fs::File::open(path.clone())?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            data.push(Doc {
                id: uuid::Uuid::new_v4().to_string(),
                content: contents,
                file_name: path.to_str().unwrap().to_string(),
            });
        }
    }
    Ok(data)
}