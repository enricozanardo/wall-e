// Dataset module - Gestione del caricamento e preprocessamento di dataset
pub mod loader;
pub mod streaming;
pub mod split;
pub mod processor;

pub use loader::{load_text, load_text_lines, load_json, load_jsonl, load_csv, DataItem};
pub use streaming::DatasetStream;
pub use split::{DatasetSplit, split_dataset, k_fold_split};
pub use processor::TextProcessor; 