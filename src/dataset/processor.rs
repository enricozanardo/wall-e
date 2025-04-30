use regex;

/// Interface for text processors
pub trait TextProcessor: Send + Sync {
    /// Processes a text string
    fn process(&self, text: &str) -> String;
}

/// Text processor that normalizes whitespace and converts to lowercase
pub struct BasicTextProcessor {
    lowercase: bool,
    normalize_whitespace: bool,
}

impl BasicTextProcessor {
    /// Creates a new basic text processor
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
            // Replace sequences of whitespace with a single space
            let re = regex::Regex::new(r"\s+").unwrap();
            processed = re.replace_all(&processed, " ").to_string();
            
            // Remove spaces at the beginning and end
            processed = processed.trim().to_string();
        }
        
        processed
    }
}

/// Processor that can execute a sequence of transformations on text
pub struct CompositeTextProcessor {
    processors: Vec<Box<dyn TextProcessor>>,
}

impl CompositeTextProcessor {
    /// Creates a new empty composite processor
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }
    
    /// Adds a processor to the chain
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