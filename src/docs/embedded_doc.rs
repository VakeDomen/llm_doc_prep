use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddedDoc {
    pub vector: Vec<f32>,
    pub content: Passage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Passage {
    usage: u32,
    text: String,
}