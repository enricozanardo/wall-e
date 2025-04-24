use crate::dataset::processor::TextProcessor;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Struttura per lo streaming di dati da un file di testo
pub struct DatasetStream {
    reader: BufReader<File>,
    batch_size: usize,
    processor: Option<Box<dyn TextProcessor>>,
    buffer: Vec<String>,
    eof_reached: bool,
}

impl DatasetStream {
    /// Crea un nuovo stream da un file
    /// 
    /// # Arguments
    /// * `path` - Percorso del file da cui leggere
    /// * `batch_size` - Dimensione del batch da leggere ad ogni chiamata a `next_batch`
    ///
    /// # Returns
    /// * `DatasetStream` - Il nuovo stream
    ///
    /// # Errors
    /// Restituisce un errore se il file non può essere aperto
    pub fn new<P: AsRef<Path>>(path: P, batch_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        Ok(Self {
            reader,
            batch_size,
            processor: None,
            buffer: Vec::with_capacity(batch_size),
            eof_reached: false,
        })
    }
    
    /// Imposta un processore di testo per lo stream
    ///
    /// # Arguments
    /// * `processor` - Il processore da utilizzare
    ///
    /// # Returns
    /// * `Self` - Lo stream con il processore impostato
    pub fn with_processor<P: TextProcessor + 'static>(mut self, processor: P) -> Self {
        self.processor = Some(Box::new(processor));
        self
    }
    
    /// Legge il prossimo batch di dati dal file
    ///
    /// # Returns
    /// * `Option<Vec<String>>` - Un batch di linee lette dal file, o None se il file è terminato
    pub fn next_batch(&mut self) -> Option<Vec<String>> {
        if self.eof_reached && self.buffer.is_empty() {
            return None;
        }
        
        // Se abbiamo ancora dati nel buffer, li restituiamo
        if !self.buffer.is_empty() {
            let batch = std::mem::take(&mut self.buffer);
            return Some(batch);
        }
        
        // Altrimenti leggiamo dal file
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF raggiunto
                    self.eof_reached = true;
                    break;
                },
                Ok(_) => {
                    // Rimuove il carattere newline se presente
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    
                    // Processa la linea se è presente un processore
                    if let Some(processor) = &self.processor {
                        line = processor.process(&line);
                    }
                    
                    batch.push(line);
                },
                Err(e) => {
                    eprintln!("Errore durante la lettura: {}", e);
                    self.eof_reached = true;
                    break;
                }
            }
        }
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
    
    /// Resetta lo stream all'inizio del file
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok se il reset è riuscito, Err altrimenti
    pub fn reset(&mut self) -> io::Result<()> {
        // Non è possibile ottenere il percorso direttamente da BufReader
        // Dovremmo memorizzare il percorso originale quando creiamo lo stream
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Reset non supportato. Creare un nuovo stream invece."
        ));
    }
}

/// Utility function per aprire uno stream di testo
pub fn open_dataset_stream<P: AsRef<Path>>(path: P, batch_size: usize) -> io::Result<DatasetStream> {
    DatasetStream::new(path, batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::processor::BasicTextProcessor;
    use std::fs;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    fn create_test_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Line 1").unwrap();
        writeln!(file, "Line 2").unwrap();
        writeln!(file, "Line 3").unwrap();
        writeln!(file, "Line 4").unwrap();
        writeln!(file, "Line 5").unwrap();
        file
    }
    
    #[test]
    fn test_dataset_stream() {
        let file = create_test_file();
        let mut stream = DatasetStream::new(file.path(), 2).unwrap();
        
        // Legge il primo batch
        let batch1 = stream.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], "Line 1");
        assert_eq!(batch1[1], "Line 2");
        
        // Legge il secondo batch
        let batch2 = stream.next_batch().unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0], "Line 3");
        assert_eq!(batch2[1], "Line 4");
        
        // Legge il terzo batch (incompleto)
        let batch3 = stream.next_batch().unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3[0], "Line 5");
        
        // Verifica che non ci siano più dati
        assert!(stream.next_batch().is_none());
    }
    
    #[test]
    fn test_stream_with_processor() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "  UPPER CASE  ").unwrap();
        writeln!(file, " Mixed   Case  ").unwrap();
        
        let processor = BasicTextProcessor::new(true, true);
        let mut stream = DatasetStream::new(file.path(), 2)
            .unwrap()
            .with_processor(processor);
        
        let batch = stream.next_batch().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], "upper case");
        assert_eq!(batch[1], "mixed case");
    }
    
    #[test]
    #[should_panic(expected = "Reset non supportato")]
    fn test_stream_reset() {
        let file = create_test_file();
        let mut stream = DatasetStream::new(file.path(), 5).unwrap();
        
        // Legge tutto il file
        let batch1 = stream.next_batch().unwrap();
        assert_eq!(batch1.len(), 5);
        assert!(stream.next_batch().is_none());
        
        // Dovrebbe fallire con errore
        stream.reset().unwrap();
    }
} 