// FUNCTION
pub const TRANSLATE: bool = false;
pub const EMBEDD: bool = true;

// TRANSLATE
pub const DOCS_TO_TRANSLATE_FOLDER: &str = "./data/a_to_translate/";
pub const TRANSLATOR_PROGRESS_FILE: &str = "./data/a_to_translate/translation_progress.json";
pub const TRANSLATOR_SYSTEM_MSG: &str = "Your task is to translate the given passages from slovene to english. The passages are given in a markdown format. You should keep the structure of the markdown and have the translation to english be as close to the original meaning as possible. It is import you only respond with the transalation and keep the markdown structure.";

// EMBEDDING
pub const QDRANT_SERVER: &str = "http://localhost:6334";
pub const QDRANT_COLLECTION: &str = "urska_md_baai_ft";

// MODELS
pub const EMBEDDING_MODEL_PATH: &str = "models/bge-large-en-v1.5-ft";
pub const EMBEDDING_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";

pub const TRANSLATION_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";
pub const TRANSLATION_MODEL: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";

// PROGRESS CONTROL
pub const FILES_TO_PROCESS: Option<usize> = None; // limiter
pub const PAR_CHUNK_SIZE: u64 = 2;


// TRANSLATION MODEL SETTINGS
pub const SYSTEM_MSG: &str = "Your task is to translate the given passages from slovene to english. The passages are given in a markdown format. You should keep the structure of the markdown and have the translation to english be as close to the original meaning as possible. It is import you only respond with the transalation and keep the markdown structure.";
pub const SEED: u64 = 42;
pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;
pub const VERBOSE_PROMPT: bool = false;
pub const SPLIT_PROPMT: bool = false;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;