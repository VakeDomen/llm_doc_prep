pub const FILES_TO_PROCESS: Option<usize> = None;
pub const PAR_CHUNK_SIZE: u64 = 2;
pub const PROGRESS_FILE: &str = "./data/progress.json";

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