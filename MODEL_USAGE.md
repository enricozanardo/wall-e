# Formato GGUF per Wall-E1

Questa guida è focalizzata sull'esportazione e l'utilizzo dei modelli Wall-E1 in formato GGUF.

## Cos'è GGUF?

GGUF (GPT-Generated Unified Format) è un formato sviluppato dal progetto `llama.cpp` per l'archiviazione efficiente di modelli di linguaggio. Il formato è progettato per essere leggero, portabile e compatibile con diverse librerie.

| Caratteristica | Dettaglio |
| --- | --- |
| Creatore | `llama.cpp` project |
| Struttura | Header + Metadata Key/Value + Tensors |
| Supporta | Tokenizer, Configurazione, Pesi |
| Compatibilità | C++, Rust, Python (`gguf-py`) |
| Versione attuale | v3 (2024) |

## Struttura interna

Un file `.gguf` è così organizzato:

1. **Header**:
   - Signature `"GGUF"` (4 bytes)
   - Version (4 bytes, intero)
   - Endianness (4 bytes, 0=little-endian, 1=big-endian)
    
2. **Meta-dati (Key-Value)**:
   - Numero di metadati (8 bytes, intero)
   - Per ogni metadato:
     - Lunghezza della chiave (8 bytes, intero)
     - Chiave (stringa UTF-8)
     - Tipo del valore (4 bytes, intero)
     - Valore (variabile in base al tipo)

3. **Tensori**:
   - Numero di tensori (8 bytes, intero)
   - Per ogni tensore:
     - Lunghezza del nome (8 bytes, intero)
     - Nome (stringa UTF-8) 
     - Numero di dimensioni (4 bytes, intero)
     - Dimensioni (8 bytes per dimensione)
     - Tipo di dato (4 bytes, intero)
     - Dati del tensore (raw bytes)

## Esportazione in formato GGUF

### Da un modello Wall-E1 esistente (usando Rust):

Il modo più semplice per esportare un modello Wall-E1 esistente in formato GGUF è utilizzare il comando:

```bash
cargo run -- --export-model path/to/model.bin --format gguf
```

Questo comando:
1. Carica il modello binario nativo specificato in `path/to/model.bin`
2. Lo converte automaticamente in formato GGUF
3. Salva il risultato nella directory `models/` mantenendo il nome del file originale ma con estensione `.gguf`

Per specificare altri formati, è possibile utilizzare le seguenti opzioni:
- `--format gguf`: Esporta in formato GGUF (per llama.cpp)
- `--format pt`: Esporta in formato PyTorch (per Python)
- `--format bin`: Salva in formato binario nativo (per Wall-E1)

### Metadati chiave inseriti durante l'esportazione

Wall-E1 inserisce i seguenti metadati essenziali:

```rust
// Metadati stringa
"general.architecture" -> "llama"
"general.name" -> "wall-e1-transformer"
"general.version" -> "1.0"
"general.description" -> "Wall-E1 Transformer Model for text generation"
"tokenizer.ggml.model" -> "llama"

// Metadati numerici
"llama.context_length" -> max_seq_len
"llama.embedding_length" -> model_dim
"llama.feed_forward_length" -> model_dim * 4
"llama.attention.head_count" -> model_dim / 64
"llama.attention.head_count_kv" -> model_dim / 64
"llama.block_count" -> params.len() / 8
"llama.rope.dimension_count" -> model_dim / 2
"tokenizer.ggml.bos_token_id" -> 1
"tokenizer.ggml.eos_token_id" -> 2
"tokenizer.ggml.unknown_token_id" -> 0
```

### Struttura dei tensori

I tensori vengono rinominati durante l'esportazione per essere compatibili con lo schema di llama.cpp:

```
"wq" -> "blk.0.attn_q.weight"
"wk" -> "blk.0.attn_k.weight"
"wv" -> "blk.0.attn_v.weight"
"wo" -> "blk.0.attn_output.weight"
"ff1" -> "blk.0.ffn_gate.weight"
"ff2" -> "blk.0.ffn_up.weight"
"ln1" -> "blk.0.attn_norm.weight"
"ln2" -> "blk.0.ffn_norm.weight"
```

Per i layer successivi, si usa `blk.N.X` dove N è il numero del layer.

## Creazione con Python

Usando la libreria `gguf-py`, è possibile creare o modificare file GGUF direttamente da Python:

```python
from gguf import GGUFWriter
import numpy as np

# Inizializzazione
writer = GGUFWriter("model.gguf", "wall-e1")

# Metadati essenziali per compatibilità
writer.add_name_value("general.architecture", "llama")
writer.add_name_value("general.name", "Wall-E1 Model")
writer.add_name_value("llama.context_length", 1024)  # dimensione contesto
writer.add_name_value("tokenizer.ggml.model", "llama")

# Tokenizer (esempio)
tokens = ["<unk>", "<s>", "</s>", "hello", "world"]
token_bytes = [t.encode("utf-8") for t in tokens]
writer.add_name_value("tokenizer.ggml.tokens", token_bytes)
writer.add_name_value("tokenizer.ggml.bos_token_id", 1)
writer.add_name_value("tokenizer.ggml.eos_token_id", 2)

# Aggiunta di tensori (esempio)
embedding = np.random.randn(len(tokens), 768).astype(np.float32)
writer.add_tensor("token_embd.weight", embedding)

query_weight = np.random.randn(768, 768).astype(np.float32)
writer.add_tensor("blk.0.attn_q.weight", query_weight)

# Finalizzazione
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

## Verifica di compatibilità

Wall-E1 fornisce uno script di verifica per controllare che il file GGUF sia ben formato e compatibile:

```bash
python scripts/verify_gguf.py path/to/model.gguf
```

Lo script esamina:
1. Header GGUF
2. Metadati richiesti
3. Presenza e organizzazione dei tensori
4. Tipi di dati utilizzati

## Caricamento di modelli GGUF in Python

```python
from scripts.load_models import WallE1GGUFLoader

# Carica il modello
loader = WallE1GGUFLoader("models/model.gguf")
metadata, tensors = loader.load()

# Visualizza informazioni
print(f"Model dimension: {metadata.get('model_dim', 0)}")
print(f"Vocab size: {metadata.get('vocab_size', 0)}")

# Accedi ai tensori (numpy arrays)
for name, tensor in tensors.items():
    print(f"Tensor {name}: shape {tensor.shape}")
```

## Utilizzo con llama.cpp

```bash
# Clona e compila llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make

# Esegui il modello
./main -m /path/to/model.gguf -n 128 -p "Ciao, come stai?"
```

## Risoluzione problemi

Se il file GGUF non viene caricato correttamente:

1. Verifica che i metadati richiesti siano presenti:
   ```bash
   python scripts/verify_gguf.py path/to/model.gguf
   ```

2. Controlla la versione del formato GGUF (v1, v2, v3)
   - Wall-E1 esporta in versione 1 per massima compatibilità

3. Verifica la mappatura dei tensori:
   ```bash
   python scripts/model_inspector.py path/to/model.gguf --type gguf
   ```

## Riferimenti

- [Specifiche GGUF](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [Libreria gguf-py](https://github.com/ggerganov/gguf-py)
- [Documentazione llama.cpp](https://github.com/ggerganov/llama.cpp) 