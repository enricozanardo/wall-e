use crate::dataset::processor::TextProcessor;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Structure for streaming data from a text file
pub struct DatasetStream {
    reader: BufReader<File>,
    batch_size: usize,
    processor: Option<Box<dyn TextProcessor>>,
    buffer: Vec<String>,
    eof_reached: bool,
}

impl DatasetStream {
    /// Creates a new stream from a file
    /// 
    /// # Arguments
    /// 
    /// * `path` - Path to the file to read from
    /// * `batch_size` - Size of the batch to read with each call to `next_batch`
    ///
    /// # Returns
    /// 
    /// * `DatasetStream` - The new stream
    ///
    /// # Errors
    /// 
    /// Returns an error if the file cannot be opened
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
    
    /// Sets a text processor for the stream
    ///
    /// # Arguments
    /// 
    /// * `processor` - The processor to use
    ///
    /// # Returns
    /// 
    /// * `Self` - The stream with the processor set
    pub fn with_processor<P: TextProcessor + 'static>(mut self, processor: P) -> Self {
        self.processor = Some(Box::new(processor));
        self
    }
    
    /// Reads the next batch of data from the file
    ///
    /// # Returns
    /// 
    /// * `Option<Vec<String>>` - A batch of lines read from the file, or None if the file is finished
    pub fn next_batch(&mut self) -> Option<Vec<String>> {
        if self.eof_reached && self.buffer.is_empty() {
            return None;
        }
        
        // If we still have data in the buffer, we return it
        if !self.buffer.is_empty() {
            let batch = std::mem::take(&mut self.buffer);
            return Some(batch);
        }
        
        // Otherwise we read from the file
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF reached
                    self.eof_reached = true;
                    break;
                },
                Ok(_) => {
                    // Remove newline character if present
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    
                    // Process the line if a processor is present
                    if let Some(processor) = &self.processor {
                        line = processor.process(&line);
                    }
                    
                    batch.push(line);
                },
                Err(e) => {
                    eprintln!("Error during reading: {}", e);
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
    
    /// Resets the stream to the beginning of the file
    ///
    /// # Returns
    /// 
    /// * `io::Result<()>` - Ok if the reset was successful, Err otherwise
    pub fn reset(&mut self) -> io::Result<()> {
        // It's not possible to get the path directly from BufReader
        // We should store the original path when we create the stream
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Reset not supported. Create a new stream instead."
        ));
    }
}

/// Utility function to open a text stream
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
        
        // Reads the first batch
        let batch1 = stream.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], "Line 1");
        assert_eq!(batch1[1], "Line 2");
        
        // Reads the second batch
        let batch2 = stream.next_batch().unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0], "Line 3");
        assert_eq!(batch2[1], "Line 4");
        
        // Reads the third batch (incomplete)
        let batch3 = stream.next_batch().unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3[0], "Line 5");
        
        // Verifies that there is no more data
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
    #[should_panic(expected = "Reset not supported")]
    fn test_stream_reset() {
        let file = create_test_file();
        let mut stream = DatasetStream::new(file.path(), 5).unwrap();
        
        // Reads the entire file
        let batch1 = stream.next_batch().unwrap();
        assert_eq!(batch1.len(), 5);
        assert!(stream.next_batch().is_none());
        
        // Should fail with error
        stream.reset().unwrap();
    }
} 