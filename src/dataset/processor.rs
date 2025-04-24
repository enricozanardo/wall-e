use rayon::prelude::*;
use std::sync::Arc;
use regex;

/// Interfaccia per il processore di testo
pub trait TextProcessor: Send + Sync {
    /// Processa una stringa di testo
    fn process(&self, text: &str) -> String;
}

/// Processore di testo che normalizza spazi bianchi e converte in minuscolo
pub struct BasicTextProcessor {
    lowercase: bool,
    normalize_whitespace: bool,
}

impl BasicTextProcessor {
    /// Crea un nuovo processore di testo basilare
    pub fn new(lowercase: bool, normalize_whitespace: bool) -> Self {
        Self {
            lowercase,
            normalize_whitespace,
        }
    }
}

impl TextProcessor for BasicTextProcessor {
    fn process(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        if self.lowercase {
            processed = processed.to_lowercase();
        }
        
        if self.normalize_whitespace {
            // Sostituisci sequenze di spazi bianchi con un singolo spazio
            let re = regex::Regex::new(r"\s+").unwrap();
            processed = re.replace_all(&processed, " ").to_string();
            
            // Rimuovi spazi all'inizio e alla fine
            processed = processed.trim().to_string();
        }
        
        processed
    }
}

/// Processore che può eseguire una sequenza di trasformazioni sul testo
pub struct CompositeTextProcessor {
    processors: Vec<Box<dyn TextProcessor>>,
}

impl CompositeTextProcessor {
    /// Crea un nuovo processore composito vuoto
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }
    
    /// Aggiunge un processore alla catena
    pub fn add_processor<P: TextProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors.push(Box::new(processor));
        self
    }
}

impl TextProcessor for CompositeTextProcessor {
    fn process(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        for processor in &self.processors {
            processed = processor.process(&processed);
        }
        
        processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_processor() {
        let processor = BasicTextProcessor::new(true, true);
        
        let input = "  Hello   WORLD!  ";
        let expected = "hello world!";
        
        assert_eq!(processor.process(input), expected);
    }
    
    #[test]
    fn test_basic_processor_no_lowercase() {
        let processor = BasicTextProcessor::new(false, true);
        
        let input = "  Hello   WORLD!  ";
        let expected = "Hello WORLD!";
        
        assert_eq!(processor.process(input), expected);
    }
    
    #[test]
    fn test_composite_processor() {
        let processor = CompositeTextProcessor::new()
            .add_processor(BasicTextProcessor::new(true, false))
            .add_processor(BasicTextProcessor::new(false, true));
        
        let input = "  Hello   WORLD!  ";
        let expected = "hello world!";
        
        assert_eq!(processor.process(input), expected);
    }
} 