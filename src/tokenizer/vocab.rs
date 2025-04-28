use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Tipo che rappresenta un vocabolario per un tokenizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocab {
    /// Mappa da token a ID
    token_to_id: HashMap<String, usize>,
    /// Mappa da ID a token
    id_to_token: HashMap<usize, String>,
    /// Set di token speciali
    special_tokens: HashSet<String>,
}

impl Vocab {
    /// Crea un nuovo vocabolario vuoto
    pub fn new() -> Self {
        Vocab {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashSet::new(),
        }
    }
    
    /// Aggiunge un token al vocabolario
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(id) = self.token_to_id.get(token) {
            return *id;
        }
        
        let id = self.token_to_id.len();
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }
    
    /// Aggiunge un token speciale al vocabolario
    pub fn add_special_token(&mut self, token: &str) -> usize {
        let id = self.add_token(token);
        self.special_tokens.insert(token.to_string());
        id
    }
    
    /// Restituisce l'ID di un token
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }
    
    /// Restituisce il token corrispondente a un ID
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    /// Verifica se un token è speciale
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains(token)
    }
    
    /// Restituisce la dimensione del vocabolario
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }
    
    /// Verifica se il vocabolario è vuoto
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
    
    /// Salva il vocabolario su file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        
        // Scrittura dei token normali
        for (token, id) in &self.token_to_id {
            if !self.special_tokens.contains(token) {
                writeln!(file, "{}\t{}", token, id)?;
            }
        }
        
        // Scrittura dei token speciali
        for token in &self.special_tokens {
            let id = self.token_to_id.get(token).unwrap();
            writeln!(file, "<special>\t{}\t{}", token, id)?;
        }
        
        Ok(())
    }
    
    /// Carica un vocabolario da file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut vocab = Vocab {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashSet::new(),
        };
        
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            
            if parts.len() == 2 {
                // Token normale
                let token = parts[0];
                let id = parts[1].parse::<usize>().unwrap_or_default();
                vocab.token_to_id.insert(token.to_string(), id);
                vocab.id_to_token.insert(id, token.to_string());
            } else if parts.len() == 3 && parts[0] == "<special>" {
                // Token speciale
                let token = parts[1];
                let id = parts[2].parse::<usize>().unwrap_or_default();
                vocab.token_to_id.insert(token.to_string(), id);
                vocab.id_to_token.insert(id, token.to_string());
                vocab.special_tokens.insert(token.to_string());
            }
        }
        
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::env::temp_dir;
    
    #[test]
    fn test_vocab_add_token() {
        let mut vocab = Vocab::new();
        
        // Aggiunta di token
        assert_eq!(vocab.add_token("hello"), 0);
        assert_eq!(vocab.add_token("world"), 1);
        
        // Verifica mappatura token -> ID
        assert_eq!(vocab.token_to_id("hello"), Some(0));
        assert_eq!(vocab.token_to_id("world"), Some(1));
        assert_eq!(vocab.token_to_id("unknown"), None);
        
        // Verifica mappatura ID -> token
        assert_eq!(vocab.id_to_token(0), Some("hello"));
        assert_eq!(vocab.id_to_token(1), Some("world"));
        assert_eq!(vocab.id_to_token(2), None);
        
        // Verifica dimensione vocabolario
        assert_eq!(vocab.len(), 2);
    }
    
    #[test]
    fn test_vocab_special_tokens() {
        let mut vocab = Vocab::new();
        
        // Aggiunta di token speciali
        assert_eq!(vocab.add_special_token("[PAD]"), 0);
        assert_eq!(vocab.add_special_token("[UNK]"), 1);
        
        // Verifica token speciali
        assert!(vocab.is_special_token("[PAD]"));
        assert!(vocab.is_special_token("[UNK]"));
        assert!(!vocab.is_special_token("hello"));
        
        // Aggiunta di token normali
        assert_eq!(vocab.add_token("hello"), 2);
        
        // Verifica dimensione vocabolario
        assert_eq!(vocab.len(), 3);
    }
    
    #[test]
    fn test_vocab_save_load() -> std::io::Result<()> {
        let mut vocab = Vocab::new();
        
        // Aggiunta di token nell'ordine corretto per il test
        vocab.add_special_token("[PAD]"); // ID 0
        vocab.add_special_token("[UNK]"); // ID 1
        vocab.add_token("hello");         // ID 2
        vocab.add_token("world");         // ID 3
        
        // Verifica ordine prima del salvataggio
        assert_eq!(vocab.token_to_id("[PAD]"), Some(0));
        assert_eq!(vocab.token_to_id("[UNK]"), Some(1));
        assert_eq!(vocab.token_to_id("hello"), Some(2));
        
        // Crea un percorso temporaneo per il file
        let mut temp_path = temp_dir();
        temp_path.push("vocab_test.txt");
        
        // Salva il vocabolario
        vocab.save(&temp_path)?;
        
        // Carica il vocabolario dal file
        let loaded_vocab = Vocab::load(&temp_path)?;
        
        // Pulisci dopo il test
        fs::remove_file(&temp_path)?;
        
        // Verifica che il vocabolario caricato sia identico all'originale
        assert_eq!(loaded_vocab.len(), vocab.len());
        assert_eq!(loaded_vocab.token_to_id("[PAD]"), vocab.token_to_id("[PAD]"));
        assert_eq!(loaded_vocab.token_to_id("[UNK]"), vocab.token_to_id("[UNK]"));
        assert_eq!(loaded_vocab.token_to_id("hello"), vocab.token_to_id("hello"));
        assert!(loaded_vocab.is_special_token("[PAD]"));
        
        Ok(())
    }
} 