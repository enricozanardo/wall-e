use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json;
use rayon::prelude::*;

/// Generic data structure for datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    /// Text content of the item
    pub text: String,
    /// Optional metadata (e.g. label, category, etc.)
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Loads an entire text file into memory
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the file to load
/// 
/// # Returns
/// 
/// * `Ok(String)` - The content of the file
/// * `Err(io::Error)` - Error during reading
pub fn load_text<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
    std::fs::read_to_string(file_path)
}

/// Loads a text file line by line, useful for very large files
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the file to load
/// 
/// # Returns
/// 
/// * `Ok(Vec<String>)` - The lines of the file
/// * `Err(io::Error)` - Error during reading
pub fn load_text_lines<P: AsRef<Path>>(file_path: P) -> io::Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    reader.lines().collect()
}

/// Loads a complete text file into memory in a parallelized way
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the file to load
/// 
/// # Returns
/// 
/// * `Ok(String)` - The content of the file
/// * `Err(io::Error)` - Error during reading
pub fn load_text_parallel<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
    // Read the lines of the file
    let lines = load_text_lines_parallel(&file_path)?;
    
    // Join the lines efficiently
    Ok(lines.join("\n"))
}

/// Loads a text file line by line in a parallelized way, useful for very large files
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the file to load
/// 
/// # Returns
/// 
/// * `Ok(Vec<String>)` - The lines of the file
/// * `Err(io::Error)` - Error during reading
pub fn load_text_lines_parallel<P: AsRef<Path>>(file_path: P) -> io::Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    // Collect the lines into a vector
    let lines: Vec<io::Result<String>> = reader.lines().collect();
    
    // Process the lines in parallel
    let result: io::Result<Vec<String>> = lines
        .into_par_iter()
        .map(|line_result| line_result.map(|line| process_line(&line)))
        .collect();
    
    result
}

/// Processes a single line of text (can be extended with more complex operations)
fn process_line(line: &str) -> String {
    // Here it's possible to add specific processing
    // For example, normalization, cleaning, tokenization, etc.
    line.to_string()
}

/// Opens a text file and returns a BufReader for streaming
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the file to load
/// 
/// # Returns
/// 
/// * `Ok(BufReader<File>)` - Reader for streaming
/// * `Err(io::Error)` - Error during opening
pub fn open_text_stream<P: AsRef<Path>>(file_path: P) -> io::Result<BufReader<File>> {
    let file = File::open(file_path)?;
    Ok(BufReader::new(file))
}

/// Loads a JSON dataset into memory
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the JSON file to load
/// 
/// # Returns
/// 
/// * `Ok(Vec<DataItem>)` - The items in the JSON file
/// * `Err(...)` - Error during reading or parsing
pub fn load_json<P: AsRef<Path>>(file_path: P) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let items: Vec<DataItem> = serde_json::from_reader(reader)?;
    Ok(items)
}

/// Loads a JSON dataset into memory using parallelization where possible
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the JSON file to load
/// 
/// # Returns
/// 
/// * `Ok(Vec<DataItem>)` - The items in the JSON file
/// * `Err(...)` - Error during reading or parsing
pub fn load_json_parallel<P: AsRef<Path>>(file_path: P) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    // First phase: load the entire JSON
    let items: Vec<DataItem> = serde_json::from_reader(reader)?;
    
    // Second phase: process the items in parallel
    let processed_items: Vec<DataItem> = items
        .into_par_iter()
        .map(|item| {
            // Here it's possible to add specific processing
            // For example, text normalization, data preparation, etc.
            item
        })
        .collect();
    
    Ok(processed_items)
}

/// Loads a JSONL (JSON Lines) dataset line by line
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the JSONL file to load
/// 
/// # Returns
/// 
/// * `Ok(Vec<DataItem>)` - The items in the JSONL file
/// * `Err(...)` - Error during reading or parsing
pub fn load_jsonl<P: AsRef<Path>>(file_path: P) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    let mut items = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let item: DataItem = serde_json::from_str(&line)?;
            items.push(item);
        }
    }
    
    Ok(items)
}

/// Loads a CSV dataset into memory
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the CSV file to load
/// * `has_headers` - Whether the file has a header row
/// * `text_column` - Name of the column containing the text (or index if there are no headers)
/// 
/// # Returns
/// 
/// * `Ok(Vec<DataItem>)` - The items extracted from the CSV
/// * `Err(...)` - Error during reading or parsing
pub fn load_csv<P: AsRef<Path>>(
    file_path: P,
    has_headers: bool,
    text_column: &str
) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    let mut csv_reader = if has_headers {
        csv::Reader::from_reader(reader)
    } else {
        csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(reader)
    };
    
    let mut items = Vec::new();
    let headers = csv_reader.headers()?.iter().collect::<Vec<_>>();
    let text_column_index = headers.iter().position(|h| *h == text_column)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Column {} not found", text_column)))?;

    for result in csv_reader.records() {
        let record = result?;
        if let Some(text) = record.get(text_column_index) {
            items.push(DataItem {
                text: text.to_string(),
                metadata: serde_json::Value::Null,
            });
        }
    }
    
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_load_text() -> io::Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "First line\nSecond line\nThird line")?;
        
        let content = load_text(file.path())?;
        assert_eq!(content, "First line\nSecond line\nThird line\n");
        
        Ok(())
    }
    
    #[test]
    fn test_load_text_lines() -> io::Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "First line")?;
        writeln!(file, "Second line")?;
        writeln!(file, "Third line")?;
        
        let lines = load_text_lines(file.path())?;
        assert_eq!(lines, vec!["First line", "Second line", "Third line"]);
        
        Ok(())
    }
    
    #[test]
    fn test_load_json() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = NamedTempFile::new()?;
        writeln!(
            file,
            r#"[
                {{"text": "Esempio 1", "metadata": {{"label": "positivo"}}}},
                {{"text": "Esempio 2", "metadata": {{"label": "negativo"}}}}
            ]"#
        )?;
        
        let items = load_json(file.path())?;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "Esempio 1");
        assert_eq!(items[1].text, "Esempio 2");
        
        Ok(())
    }
    
    #[test]
    fn test_load_jsonl() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, r#"{{"text": "Esempio 1", "metadata": {{"label": "positivo"}}}}"#)?;
        writeln!(file, r#"{{"text": "Esempio 2", "metadata": {{"label": "negativo"}}}}"#)?;
        
        let items = load_jsonl(file.path())?;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "Esempio 1");
        assert_eq!(items[1].text, "Esempio 2");
        
        Ok(())
    }
    
    #[test]
    fn test_load_csv() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "testo,label")?;
        writeln!(file, "\"Esempio 1\",positivo")?;
        writeln!(file, "\"Esempio 2\",negativo")?;
        
        let items = load_csv(file.path(), true, "testo")?;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "Esempio 1");
        assert_eq!(items[1].text, "Esempio 2");
        
        Ok(())
    }
} 