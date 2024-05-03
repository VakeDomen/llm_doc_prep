// FUNCTION
pub const TRANSLATE: bool = false;
pub const EMBEDD: bool = false;
pub const DECORATE: bool = true;
pub const GENERATE_EMBEDDING_QUESTIONS: bool = true;

// TRANSLATE
pub const DOCS_TO_TRANSLATE_FOLDER: &str = "./data/a_to_translate/";
pub const TRANSLATOR_PROGRESS_FILE: &str = "./data/a_to_translate/translation_progress.json";
pub const TRANSLATOR_SYSTEM_MSG: &str = "Your task is to translate the given passages from slovene to english. The passages are given in a markdown format. You should keep the structure of the markdown and have the translation to english be as close to the original meaning as possible. It is import you only respond with the transalation and keep the markdown structure.";


// EMBEDDING
pub const DOCS_TO_EMBEDD_FOLDER: &str = "./data/processed/";
pub const KEYWORD_DECORATOR_PROGRESS_FILE: &str = "./data/processed/decoration_progress.json";
pub const QDRANT_SERVER: &str = "http://localhost:6334";
pub const QDRANT_COLLECTION: &str = "urska_md_baai_ft_decorated";
pub const KEYWORD_DECORATOR_SYSTEM_MSG: &str = "Your task is to generate an unordered list of keywords about a given text passage. The passages are given in a markdown format. The passages are part of documents and information about University of Primorska. The keywords should cover what the passage is talking about. Generate up to 5 keywords. If applicable the study programm should be on the list of keywords. For clues you are also given the name of the document that the passage was taken from. The keywords should be generated from the perspective of what the document would mean to the student. It is important you only respond with keywords.";

// MODELS
pub const EMBEDDING_MODEL_PATH: &str = "models/bge-large-en-v1.5-ft";
pub const EMBEDDING_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";

pub const TRANSLATION_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";
pub const TRANSLATION_MODEL: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";

pub const QUESTION_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";
pub const QUESTION_MODEL: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";

pub const KEYWORD_DECORATOR_TOKENIZER: &str = "models/llama3-8b/tokenizer.json";
pub const KEYWORD_DECORATOR_MODEL: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";

// PROGRESS CONTROL
pub const FILES_TO_PROCESS: Option<usize> = None; // limiter
pub const PAR_CHUNK_SIZE: u64 = 2;


// TRANSLATION MODEL SETTINGS
pub const SEED: u64 = 42;
pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;
pub const VERBOSE_PROMPT: bool = false;
pub const SPLIT_PROPMT: bool = false;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;