use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Doc {
    pub id: String,
    pub content: String,
    pub file_name: String,
}