# Dataset per Wall-E1

Questa cartella contiene i dataset utilizzati per l'addestramento e il testing dei modelli in Wall-E1.

## Formati supportati

Il sistema supporta i seguenti formati di dataset:

### File di testo (.txt)

I file di testo semplici possono essere utilizzati per l'addestramento e il testing. Viene utilizzato un file separato per ogni insieme (training e test).

Esempio:
- `story.txt` - Testo di addestramento
- `test.txt` - Testo di testing

### File JSON (.json)

I file JSON consentono di specificare sia i dati di addestramento/testing che i parametri del modello in un unico file.

Formato:
```json
{
  "train_text": "Testo per l'addestramento...",
  "test_text": "Testo per il testing...",
  "prompts": [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
  ],
  "model_params": {
    "d_model": 64,
    "max_seq_len": 128,
    "num_heads": 4,
    "ff_dim": 128,
    "num_layers": 2,
    "dropout_rate": 0.1,
    "learning_rate": 0.001
  }
}
```

Parametri:
- `train_text`: Il testo utilizzato per l'addestramento del modello (obbligatorio)
- `test_text`: Il testo utilizzato per valutare il modello (obbligatorio)
- `prompts`: Array di stringhe utilizzate come prompts per la generazione di testo (opzionale)
- `model_params`: Parametri del modello (opzionale)
  - `d_model`: Dimensione degli embedding (default: 64)
  - `max_seq_len`: Lunghezza massima delle sequenze (default: 128)
  - `num_heads`: Numero di teste nell'attenzione multi-testa (default: 4)
  - `ff_dim`: Dimensione interna del feed-forward network (default: 128)
  - `num_layers`: Numero di layer nell'encoder stack (default: 2)
  - `dropout_rate`: Tasso di dropout (default: 0.1)
  - `learning_rate`: Tasso di apprendimento (default: 0.001)

## Utilizzo

Per avviare l'addestramento con un file di configurazione JSON specifico:

```bash
cargo run -- data/dataset.json
```

Se non viene specificato alcun percorso, verrà utilizzato il file predefinito `data/dataset.json`.

## Dataset disponibili

- `dataset.json` - Un file JSON di esempio che contiene un estratto di storia
- `story.txt` - Un file di testo di esempio che contiene un estratto di storia
- `test.txt` - Un file di testo di esempio per il testing

## Aggiungere nuovi dataset

È possibile aggiungere nuovi dataset in questa cartella creando nuovi file di testo o JSON. Assicurarsi che i file JSON seguano il formato descritto sopra.

Per dataset più grandi, è consigliabile utilizzare il modulo `DatasetStream` per lo streaming efficiente dei dati:

```rust
use wall_e1::dataset::streaming::DatasetStream;
use wall_e1::dataset::processor::BasicTextProcessor;

// Crea uno stream di dataset con batch di 64 elementi
let mut stream = DatasetStream::new("path/to/large_dataset.txt", 64)
    .with_processor(BasicTextProcessor::new(true, true));

// Leggi i batch uno per uno
while let Some(batch) = stream.next_batch() {
    // Elabora il batch
    for text in batch {
        println!("Testo: {}", text);
    }
}
``` 