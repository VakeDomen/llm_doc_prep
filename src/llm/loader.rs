use std::path::Path;

use anyhow::{Error, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

use crate::config::EMBEDDING_MODEL_PATH;

pub type LoadedEmbeddingModel = (BertModel, Tokenizer, Device);

/// Loads a BERT embedding model along with its tokenizer from disk, targeting a specified GPU device.
///
/// This function initializes a device according to a provided `gpu_id`, then attempts to load both a BERT model
/// and a tokenizer from predefined paths. The model and tokenizer are essential for generating embeddings for NLP tasks.
///
/// # Parameters
/// - `gpu_id`: An optional identifier for a GPU device. If `None`, the model loads on the default device.
///
/// # Returns
/// Returns a `Result` containing a tuple of the loaded model, tokenizer, and device if successful, or panics if
/// any loading step fails.
///
/// # Panics
/// - Panics if the BERT model or tokenizer cannot be loaded from the specified paths. This is typically due to file path issues
///   or incorrect model/tokenizer configurations.
pub fn load_bert_model(gpu_id: Option<usize>) -> Result<LoadedEmbeddingModel> {
    let device = load_device(gpu_id);
    let model = match load_pybin_bert_model_from_disk(EMBEDDING_MODEL_PATH, &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load embedding model: {:#?}", e),
    };
    let tokenizer = match load_tokenizer(&format!("{}/tokenizer.json", EMBEDDING_MODEL_PATH)) {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };
    Ok((model, tokenizer, device))
}


/// Initializes a computational device based on the provided GPU identifier.
///
/// This function selects a computational device for running operations. If a valid GPU ID is provided
/// and the system supports CUDA, it will attempt to initialize a CUDA device. If the initialization fails,
/// or if no GPU ID is provided, it defaults to using the CPU.
///
/// # Arguments
/// * gpu_id - An optional GPU identifier for attempting to initialize a CUDA device.
/// If None, or if CUDA initialization fails, the CPU is used.
///
/// # Returns
/// Returns a Device enum, which could be either Device::Cpu or Device::Cuda.
///
/// # Notes
/// * The function handles errors internally by logging them and falling back to CPU usage.
/// * This approach ensures that the application can continue running even if CUDA is not available.
fn load_device(gpu_id: Option<usize>) -> Device {
    if let Some(id) = gpu_id {
        match Device::new_cuda(id) {
            Ok(cuda) => cuda,
            Err(e) => {
                println!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
                Device::Cpu
            },
        }
    } else {
        Device::Cpu
    }
}


/// Loads a BERT model from disk using a specified device configuration.
///
/// This function reads the model configuration from a JSON file and the binary weights from a PyTorch `.bin` file.
/// It handles the parsing and loading of the model into a specified computational device, suitable for subsequent NLP tasks.
///
/// # Parameters
/// - `model_path`: The path to the directory containing the model's configuration and binary files.
/// - `device`: A reference to the `Device` configuration indicating where the model should be loaded (e.g., CPU, GPU).
///
/// # Returns
/// Returns a `Result` containing the `BertModel` if successfully loaded, or an error if the loading process fails at any step.
///
/// # Errors
/// - If the configuration file cannot be read, an error is logged and returned.
/// - If the configuration JSON cannot be parsed, an error is logged and returned.
/// - If the binary model file cannot be loaded into the `VarBuilder`, an error is logged and returned.
fn load_pybin_bert_model_from_disk(model_path: &str, device: &Device) -> Result<BertModel> {
    let config = match std::fs::read_to_string(format!("{}/config.json", model_path)) {
        Ok(c) => c,
        Err(e) => {
            println!("Failed loading embedding model config from file: {:#?}", e);
            return Err(e.into());
        },
    };
    let config: Config = match serde_json::from_str(&config){
        Ok(c) => c,
        Err(e) => {
            println!("Failed parsing embedding model config from JSON string: {:#?}", e);
            return Err(e.into());
        },
    };
    let model_path_string = format!("{}/pytorch_model.bin", model_path);
    let model_path = Path::new(&model_path_string);
    let vb = match VarBuilder::from_pth(model_path, DTYPE, device){
        Ok(c) => c,
        Err(e) => {
            println!("Failed parsing VarBuilder from PytorchBin model path: {:#?}", e);
            return Err(e.into());
        },
    };
    Ok(BertModel::load(vb, &config)?)
}



/// Loads a `Tokenizer` from a specified file path.
/// 
/// # Arguments
/// * `tokenizer_path` - A string slice that holds the file path to the tokenizer model.
///
/// # Returns
/// A `Result` which, on success, contains the `Tokenizer`, and on failure, contains an `Error`.
fn load_tokenizer(tokenizer_path: &str) -> Result<Tokenizer> {
    let tokenizer_path = std::path::PathBuf::from(tokenizer_path);
    Tokenizer::from_file(tokenizer_path).map_err(Error::msg)
}