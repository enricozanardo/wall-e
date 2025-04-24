use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use rayon::prelude::*;

/// Struttura dati generica per dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    /// Testo dell'elemento
    pub text: String,
    /// Metadati opzionali (ad es. etichetta, categoria, ecc.)
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Carica un file di testo completo in memoria
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file da caricare
/// 
/// # Restituisce
/// * `Ok(String)` - Il contenuto del file
/// * `Err(io::Error)` - Errore durante la lettura
pub fn load_text<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
    std::fs::read_to_string(file_path)
}

/// Carica un file di testo linea per linea, utile per file molto grandi
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file da caricare
/// 
/// # Restituisce
/// * `Ok(Vec<String>)` - Le righe del file
/// * `Err(io::Error)` - Errore durante la lettura
pub fn load_text_lines<P: AsRef<Path>>(file_path: P) -> io::Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    reader.lines().collect()
}

/// Carica un file di testo completo in memoria in modo parallelizzato
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file da caricare
/// 
/// # Restituisce
/// * `Ok(String)` - Il contenuto del file
/// * `Err(io::Error)` - Errore durante la lettura
pub fn load_text_parallel<P: AsRef<Path>>(file_path: P) -> io::Result<String> {
    // Legge le linee del file
    let lines = load_text_lines_parallel(&file_path)?;
    
    // Unisce le linee in modo efficiente
    Ok(lines.join("\n"))
}

/// Carica un file di testo linea per linea in modo parallelizzato, utile per file molto grandi
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file da caricare
/// 
/// # Restituisce
/// * `Ok(Vec<String>)` - Le righe del file
/// * `Err(io::Error)` - Errore durante la lettura
pub fn load_text_lines_parallel<P: AsRef<Path>>(file_path: P) -> io::Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    // Raccoglie le linee in un vettore
    let lines: Vec<io::Result<String>> = reader.lines().collect();
    
    // Processa le linee in parallelo
    let result: io::Result<Vec<String>> = lines
        .into_par_iter()
        .map(|line_result| line_result.map(|line| process_line(&line)))
        .collect();
    
    result
}

/// Elabora una singola linea di testo (può essere estesa con operazioni più complesse)
fn process_line(line: &str) -> String {
    // Qui è possibile aggiungere elaborazione specifica
    // Ad esempio, normalizzazione, pulizia, tokenizzazione, ecc.
    line.to_string()
}

/// Apre un file di testo e restituisce un BufReader per lo streaming
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file da caricare
/// 
/// # Restituisce
/// * `Ok(BufReader<File>)` - Reader per lo streaming
/// * `Err(io::Error)` - Errore durante l'apertura
pub fn open_text_stream<P: AsRef<Path>>(file_path: P) -> io::Result<BufReader<File>> {
    let file = File::open(file_path)?;
    Ok(BufReader::new(file))
}

/// Carica un dataset JSON in memoria
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file JSON da caricare
/// 
/// # Restituisce
/// * `Ok(Vec<DataItem>)` - Gli elementi nel file JSON
/// * `Err(...)` - Errore durante la lettura o il parsing
pub fn load_json<P: AsRef<Path>>(file_path: P) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let items: Vec<DataItem> = serde_json::from_reader(reader)?;
    Ok(items)
}

/// Carica un dataset JSON in memoria utilizzando parallelizzazione dove possibile
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file JSON da caricare
/// 
/// # Restituisce
/// * `Ok(Vec<DataItem>)` - Gli elementi nel file JSON
/// * `Err(...)` - Errore durante la lettura o il parsing
pub fn load_json_parallel<P: AsRef<Path>>(file_path: P) -> Result<Vec<DataItem>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    // Prima fase: carica l'intero JSON
    let items: Vec<DataItem> = serde_json::from_reader(reader)?;
    
    // Seconda fase: processa gli elementi in parallelo
    let processed_items: Vec<DataItem> = items
        .into_par_iter()
        .map(|item| {
            // Qui è possibile aggiungere elaborazione specifica
            // Ad esempio, normalizzazione del testo, preparazione dei dati, ecc.
            item
        })
        .collect();
    
    Ok(processed_items)
}

/// Carica un dataset JSONL (JSON Lines) linea per linea
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file JSONL da caricare
/// 
/// # Restituisce
/// * `Ok(Vec<DataItem>)` - Gli elementi nel file JSONL
/// * `Err(...)` - Errore durante la lettura o il parsing
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

/// Carica un dataset CSV in memoria
/// 
/// # Argomenti
/// * `file_path` - Il percorso del file CSV da caricare
/// * `has_headers` - Se il file ha una riga di intestazione
/// * `text_column` - Nome della colonna contenente il testo (o indice se non ci sono intestazioni)
/// 
/// # Restituisce
/// * `Ok(Vec<DataItem>)` - Gli elementi estratti dal CSV
/// * `Err(...)` - Errore durante la lettura o il parsing
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
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Colonna {} non trovata", text_column)))?;

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
        writeln!(file, "Prima riga\nSeconda riga\nTerza riga")?;
        
        let content = load_text(file.path())?;
        assert_eq!(content, "Prima riga\nSeconda riga\nTerza riga\n");
        
        Ok(())
    }
    
    #[test]
    fn test_load_text_lines() -> io::Result<()> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "Prima riga")?;
        writeln!(file, "Seconda riga")?;
        writeln!(file, "Terza riga")?;
        
        let lines = load_text_lines(file.path())?;
        assert_eq!(lines, vec!["Prima riga", "Seconda riga", "Terza riga"]);
        
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