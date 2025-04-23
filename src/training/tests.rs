use crate::training::{CrossEntropyLoss, AdamOptimizer, Trainer, ModelOutput, evaluate};
use crate::nabla::tensor::Tensor;
use crate::tokenizer::basic_tokenizer::BasicTokenizer;
use ndarray::{Array2, Array3};
use std::collections::HashMap;

#[test]
fn test_cross_entropy_loss_basic() {
    let loss_fn = CrossEntropyLoss::new();
    
    // Crea logits semplici: batch_size=1, seq_len=1, vocab_size=3
    let logits_data = Array3::<f32>::from_shape_vec(
        (1, 1, 3),
        vec![1.0, 2.0, 0.5]
    ).unwrap();
    
    let logits = Tensor::new_3d(logits_data);
    
    // Target: il secondo token (indice 1) è quello corretto
    let targets = Array2::<usize>::from_shape_vec(
        (1, 1),
        vec![1]
    ).unwrap();
    
    // Calcola loss e gradiente
    let (loss, grad) = loss_fn.forward(&logits, &targets, None);
    
    // Verifica che la loss sia positiva
    assert!(loss > 0.0, "La loss dovrebbe essere positiva");
    
    // Il gradiente dovrebbe avere la forma dei logits
    assert_eq!(grad.data.shape(), logits.data.shape());
    
    // Calcola softmax manualmente per confronto
    let max_logit = 2.0_f32; // Il valore massimo nei logits
    let exp_logits = vec![
        (1.0_f32 - max_logit).exp(),
        (2.0_f32 - max_logit).exp(),
        (0.5_f32 - max_logit).exp()
    ];
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();
    
    // Il gradiente del token corretto (indice 1) dovrebbe essere (prob - 1.0)
    let grad_value = grad.data.get([0, 0, 1]).unwrap();
    assert!(
        (grad_value - (probs[1] - 1.0)).abs() < 1e-5,
        "Gradiente errato per il token corretto"
    );
    
    // Gli altri gradienti dovrebbero essere uguali alle probabilità
    assert!(
        (grad.data.get([0, 0, 0]).unwrap() - probs[0]).abs() < 1e-5,
        "Gradiente errato per token non bersaglio"
    );
    assert!(
        (grad.data.get([0, 0, 2]).unwrap() - probs[2]).abs() < 1e-5,
        "Gradiente errato per token non bersaglio"
    );
}

#[test]
fn test_cross_entropy_loss_with_padding() {
    let loss_fn = CrossEntropyLoss::new();
    
    // Crea logits con padding: batch_size=2, seq_len=3, vocab_size=4
    let logits_data = Array3::<f32>::from_shape_vec(
        (2, 3, 4),
        vec![
            // Batch 0
            1.0, 2.0, 0.5, 1.5,  // Token 0
            1.0, 2.0, 0.5, 1.5,  // Token 1
            1.0, 2.0, 0.5, 1.5,  // Token 2 (padding)
            
            // Batch 1
            1.0, 2.0, 0.5, 1.5,  // Token 0
            1.0, 2.0, 0.5, 1.5,  // Token 1
            1.0, 2.0, 0.5, 1.5,  // Token 2
        ]
    ).unwrap();
    
    let logits = Tensor::new_3d(logits_data);
    
    // Target con padding (0 è il token di padding)
    let targets = Array2::<usize>::from_shape_vec(
        (2, 3),
        vec![
            // Batch 0
            1, 2, 0,  // L'ultimo è padding
            // Batch 1
            3, 1, 2,  // Nessun padding
        ]
    ).unwrap();
    
    // Calcola loss ignorando l'indice di padding 0
    let (loss_with_ignore, grad_with_ignore) = loss_fn.forward(&logits, &targets, Some(0));
    
    // Calcola loss senza ignorare l'indice di padding
    let (loss_without_ignore, _) = loss_fn.forward(&logits, &targets, None);
    
    // La loss senza ignorare dovrebbe essere diversa
    assert!(
        (loss_with_ignore - loss_without_ignore).abs() > 1e-5,
        "Le loss dovrebbero essere diverse quando si ignora il padding"
    );
    
    // Il gradiente per il token di padding dovrebbe essere zero
    let mut sum = 0.0;
    for v in 0..4 {
        if let Some(&val) = grad_with_ignore.data.get([0, 2, v]) {
            sum += val;
        }
    }
    assert!(
        sum.abs() < 1e-5,
        "Il gradiente per il token di padding dovrebbe essere zero"
    );
}

#[test]
fn test_adam_optimizer_basic() {
    // Crea un ottimizzatore Adam
    let mut optimizer = AdamOptimizer::new(0.1, 0.9, 0.999, 1e-8);
    
    // Crea un parametro
    let param_data = ndarray::Array2::<f32>::zeros((2, 2));
    let param = Tensor::new(param_data);
    
    // Crea un gradiente unitario
    let grad_data = ndarray::Array2::<f32>::ones((2, 2));
    let grad = Tensor::new(grad_data);
    
    // Aggiungi parametro e gradiente
    let mut params = HashMap::new();
    params.insert("w".to_string(), param);
    
    let mut grads = HashMap::new();
    grads.insert("w".to_string(), grad);
    
    // Esegui un passo di ottimizzazione
    optimizer.step(&mut params, &grads);
    
    // Verifica che i parametri siano cambiati
    let updated_param = params.get("w").unwrap();
    
    // Il parametro dovrebbe essere aggiornato in direzione negativa del gradiente
    assert!(updated_param.data[[0, 0]] < 0.0, "Il parametro dovrebbe diminuire");
    
    // Il valore dovrebbe essere circa -0.1 (learning rate)
    // Ma con fattori di correzione Adam
    assert!(
        (updated_param.data[[0, 0]] + 0.1).abs() < 0.01,
        "L'aggiornamento non è della dimensione attesa"
    );
}

#[test]
fn test_adam_optimizer_momentum() {
    // Crea un ottimizzatore Adam
    let mut optimizer = AdamOptimizer::new(0.1, 0.9, 0.999, 1e-8);
    
    // Crea un parametro
    let param_data = ndarray::Array2::<f32>::zeros((1, 1));
    let param = Tensor::new(param_data);
    
    // Aggiungi parametro
    let mut params = HashMap::new();
    params.insert("w".to_string(), param);
    
    // Prima iterazione: gradiente positivo
    let grad_pos = Tensor::new(ndarray::Array2::<f32>::from_elem((1, 1), 1.0));
    let mut grads_pos = HashMap::new();
    grads_pos.insert("w".to_string(), grad_pos);
    optimizer.step(&mut params, &grads_pos);
    
    let _after_pos = params.get("w").unwrap().data[[0, 0]];
    
    // Seconda iterazione: gradiente negativo
    let grad_neg = Tensor::new(ndarray::Array2::<f32>::from_elem((1, 1), -1.0));
    let mut grads_neg = HashMap::new();
    grads_neg.insert("w".to_string(), grad_neg);
    optimizer.step(&mut params, &grads_neg);
    
    let after_neg = params.get("w").unwrap().data[[0, 0]];
    
    // Il momento dovrebbe smorzare l'effetto del cambiamento di segno
    assert!(
        after_neg < 0.0,
        "L'effetto del momento dovrebbe mantenere la direzione negativa"
    );
}

#[test]
fn test_trainer_initialization() {
    // Crea un tokenizer di base
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab("Questo è un test di un trainer model", 1);
    
    // Parametri del modello
    let model_dim = 8;
    let ff_dim = 16;
    let num_heads = 2;
    let num_layers = 1;
    let dropout_rate = 0.1;
    let learning_rate = 0.001;
    
    // Crea il trainer
    let trainer = Trainer::new(
        Box::new(tokenizer),
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Verifica che la struttura sia stata inizializzata correttamente
    assert_eq!(trainer.model_dim, model_dim);
    
    // Verifica che l'output projection sia stato inizializzato
    assert!(trainer.params.contains_key("output_projection"));
}

#[test]
fn test_trainer_forward_shape() {
    // Crea un tokenizer minimale
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab("a b c", 1);
    
    let vocab_size = 5; // Token speciali ([PAD], [UNK]) + (a, b, c)
    
    // Parametri del modello
    let model_dim = 8;
    let ff_dim = 16;
    let num_heads = 2;
    let num_layers = 1;
    let dropout_rate = 0.0; // Disattiva il dropout per i test
    let learning_rate = 0.001;
    
    // Crea il trainer
    let trainer = Trainer::new(
        Box::new(tokenizer),
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Crea un batch con una sequenza
    let batch = vec![vec![1, 2, 3]];
    
    // Forward pass
    let output = trainer.forward(&batch, None);
    
    // Verifica le dimensioni dell'output
    let shape = output.logits.data.shape();
    assert_eq!(shape.len(), 3, "L'output dovrebbe essere un tensore 3D");
    assert_eq!(shape[0], 1, "Batch size non corretto");
    assert_eq!(shape[1], 3, "Lunghezza della sequenza non corretta");
    assert_eq!(shape[2], vocab_size, "Dimensione del vocabolario non corretta");
}

#[test]
fn test_trainer_train_step() {
    // Crea un tokenizer minimale
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab("a b c", 1);
    
    // Parametri del modello
    let model_dim = 8;
    let ff_dim = 16;
    let num_heads = 2;
    let num_layers = 1;
    let dropout_rate = 0.0; // Disattiva il dropout per i test
    let learning_rate = 0.01; // Learning rate più alto per test
    
    // Crea il trainer
    let mut trainer = Trainer::new(
        Box::new(tokenizer),
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Salva i parametri originali
    let original_output_proj = trainer.output_projection.data.clone();
    
    // Crea un batch con una sequenza
    let batch = vec![vec![1, 2]];
    
    // Crea target
    let targets = Array2::<usize>::from_shape_vec(
        (1, 2),
        vec![2, 3]
    ).unwrap();
    
    // Esegui un passo di training
    let loss = trainer.train_step(&batch, &targets);
    
    // Verifica che la loss sia positiva
    assert!(loss > 0.0, "La loss dovrebbe essere positiva");
    
    // Verifica che i parametri siano cambiati
    assert!(
        (trainer.output_projection.data.clone() - original_output_proj)
            .mapv(|x| x.abs())
            .sum() > 0.0,
        "I parametri dovrebbero essere aggiornati dopo il train_step"
    );
} 