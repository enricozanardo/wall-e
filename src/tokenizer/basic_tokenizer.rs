use std::collections::HashMap;
use crate::tokenizer::{Tokenizer, Vocab};

/// Tokenizer di base che utilizza spazi e punteggiatura per dividere il testo
#[derive(Debug, Clone)]
pub struct BasicTokenizer {
    vocab: Vocab,
    unk_token: String,
}

impl BasicTokenizer {
    /// Crea un nuovo tokenizer di base con vocabolario vuoto
    pub fn new() -> Self {
        let mut vocab = Vocab::new();
        
        // Aggiungi token speciali di default
        vocab.add_special_token("[PAD]"); // Padding token
        let _unk_id = vocab.add_special_token("[UNK]"); // Unknown token
        
        BasicTokenizer {
            vocab,
            unk_token: "[UNK]".to_string(),
        }
    }
    
    /// Crea un nuovo tokenizer di base con vocabolario predefinito
    pub fn with_vocab(vocab: Vocab) -> Self {
        // Assicurati che il token unknown esista
        let mut vocab = vocab;
        if vocab.token_to_id("[UNK]").is_none() {
            vocab.add_special_token("[UNK]");
        }
        
        BasicTokenizer {
            vocab,
            unk_token: "[UNK]".to_string(),
        }
    }
    
    /// Costruisce un vocabolario a partire da un testo
    pub fn build_vocab(&mut self, text: &str, min_freq: usize) {
        // Tokenizza il testo
        let tokens = self.tokenize_raw(text);
        
        // Conteggio delle frequenze
        let mut freqs: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *freqs.entry(token).or_insert(0) += 1;
        }
        
        // Aggiungi i token che superano la frequenza minima
        for (token, freq) in freqs {
            if freq >= min_freq {
                self.vocab.add_token(&token);
            }
        }
    }
    
    /// Tokenizza il testo senza utilizzare il vocabolario (per costruzione del vocabolario)
    fn tokenize_raw(&self, text: &str) -> Vec<String> {
        // Semplice tokenizzazione basata su spazi e punteggiatura
        let mut tokens = Vec::new();
        
        // Sostituisci la punteggiatura con spazi + punteggiatura + spazi
        let text = text.replace('.', " . ")
                       .replace(',', " , ")
                       .replace('!', " ! ")
                       .replace('?', " ? ")
                       .replace(':', " : ")
                       .replace(';', " ; ")
                       .replace('(', " ( ")
                       .replace(')', " ) ")
                       .replace('[', " [ ")
                       .replace(']', " ] ")
                       .replace('{', " { ")
                       .replace('}', " } ");
        
        // Dividi per spazi
        for token in text.split_whitespace() {
            if !token.is_empty() {
                tokens.push(token.to_lowercase());
            }
        }
        
        tokens
    }
    
    /// Ottieni il vocabolario
    pub fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    /// Ottieni una referenza mutabile al vocabolario
    pub fn get_vocab_mut(&mut self) -> &mut Vocab {
        &mut self.vocab
    }
}

impl Tokenizer for BasicTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_raw(text)
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let unk_id = self.vocab.token_to_id(&self.unk_token).unwrap();
        
        tokens.iter()
              .map(|token| self.vocab.token_to_id(token).unwrap_or(unk_id))
              .collect()
    }
    
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
           .filter_map(|&id| self.vocab.id_to_token(id))
           .collect::<Vec<_>>()
           .join(" ")
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenizer_tokenize() {
        let tokenizer = BasicTokenizer::new();
        
        let text = "Hello, world!";
        let tokens = tokenizer.tokenize(text);
        
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }
    
    #[test]
    fn test_basic_tokenizer_encode_decode() {
        let mut tokenizer = BasicTokenizer::new();
        
        // Costruisci un piccolo vocabolario
        tokenizer.get_vocab_mut().add_token("hello");
        tokenizer.get_vocab_mut().add_token("world");
        
        let text = "Hello, world!";
        let ids = tokenizer.encode(text);
        
        // "hello" e "world" devono essere nel vocabolario, "," e "!" sono [UNK]
        let unk_id = tokenizer.get_vocab().token_to_id("[UNK]").unwrap();
        assert_eq!(ids[0], tokenizer.get_vocab().token_to_id("hello").unwrap());
        assert_eq!(ids[1], unk_id); // ","
        assert_eq!(ids[2], tokenizer.get_vocab().token_to_id("world").unwrap());
        assert_eq!(ids[3], unk_id); // "!"
        
        // Decodifica (nota che perdiamo la punteggiatura)
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello [UNK] world [UNK]");
    }
    
    #[test]
    fn test_build_vocab() {
        let mut tokenizer = BasicTokenizer::new();
        
        let text = "hello world hello hello world test";
        tokenizer.build_vocab(text, 2); // Solo token con frequenza >= 2
        
        // "hello" e "world" sono abbastanza frequenti
        assert!(tokenizer.get_vocab().token_to_id("hello").is_some());
        assert!(tokenizer.get_vocab().token_to_id("world").is_some());
        
        // "test" non è abbastanza frequente
        assert!(tokenizer.get_vocab().token_to_id("test").is_none());
    }
} 