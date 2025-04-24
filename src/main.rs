mod nabla;
mod tokenizer;
mod embedding;
mod attention;
mod training;

use ndarray::{Array, Array3};
use nabla::tensor::Tensor;
use tokenizer::{Tokenizer, basic_tokenizer::BasicTokenizer};
use embedding::TransformerEmbedding;
use attention::{Attention, SelfAttention, MultiHeadAttention, FeedForward};
use training::Trainer;


fn main() {
    println!("Hello, world!");
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
      
    // Esempio di training di un modello semplice
    // train_simple_model();
    
    // Esempio di utilizzo di tokenizer e embedding
    embedding_example();
    
    // Esempio completo di transformer pipeline
    transformer_pipeline_example();
    
    // Esempio di training di un modello transformer
    training_example();
}

// Esempio che dimostra la pipeline completa da una frase all'output del transformer
fn transformer_pipeline_example() {
    println!("\n--- Transformer Pipeline Example ---");
    
    // 1. Input
    let input_text = "Io sono un robot";
    println!("Input text: '{}'", input_text);
    
    // 2. Tokenizzazione
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab(input_text, 1); // Costruisce il vocabolario
    
    // Aggiungiamo altre parole per arricchire il vocabolario
    tokenizer.build_vocab("Ciao mondo come stai intelligenza artificiale", 1);
    
    let token_ids = tokenizer.encode(input_text);
    println!("Token IDs: {:?}", token_ids);
    
    // 3. Configurazione dei parametri del modello
    let vocab_size = tokenizer.get_vocab().len();
    let d_model = 64;      // Dimensione dell'embedding
    let max_seq_len = 128; // Lunghezza massima delle sequenze
    let num_heads = 4;     // Numero di teste per multi-head attention
    let ff_dim = 128;      // Dimensione interna del feed-forward network
    let num_layers = 2;    // Numero di layer nell'encoder stack
    let dropout_rate = 0.1;
    
    println!("Model config: d_model={}, num_heads={}, ff_dim={}, num_layers={}, vocab_size={}", 
             d_model, num_heads, ff_dim, num_layers, vocab_size);
    
    // 4. Creare l'embedding layer
    let embedding = TransformerEmbedding::new(
        vocab_size,
        d_model,
        max_seq_len,
        dropout_rate
    );
    
    // 5. Forward pass attraverso l'embedding layer
    let embedded = embedding.forward(&token_ids);
    println!("Embedding output shape: {:?}", embedded.data.shape());
    
    // Convertiamo l'output dell'embedding in formato adatto per self-attention
    // Reshaping da [seq_len, d_model] a [batch=1, seq_len, d_model]
    let seq_len = token_ids.len();
    let mut embedded_3d = Array3::<f32>::zeros((1, seq_len, d_model));
    
    for i in 0..seq_len {
        for j in 0..d_model {
            embedded_3d[[0, i, j]] = embedded.data[[i, j]];
        }
    }
    
    let embedded_tensor = Tensor::new_3d(embedded_3d);
    
    // 6. Self-Attention
    println!("\nProcessing through Self-Attention...");
    let self_attention = SelfAttention::new(d_model, 0.1);
    
    // Self-attention richiede Q, K, V identici per il self-attention puro
    let attention_output = self_attention.forward(
        &embedded_tensor, 
        &embedded_tensor, 
        &embedded_tensor, 
        None  // No mask
    );
    
    println!("Self-Attention output shape: {:?}", attention_output.data.shape());
    
    // 7. Multi-Head Attention
    println!("\nProcessing through Multi-Head Attention...");
    let multi_head = MultiHeadAttention::new(d_model, num_heads, 0.1);
    
    let mha_output = multi_head.forward(
        &embedded_tensor,
        &embedded_tensor,
        &embedded_tensor,
        None  // No mask
    );
    
    println!("Multi-Head Attention output shape: {:?}", mha_output.data.shape());
    
    // 8. Feed-Forward Network
    println!("\nProcessing through Feed-Forward Network...");
    
    // Reshape da [batch, seq_len, d_model] a [batch*seq_len, d_model] per FFN
    let batch_size = mha_output.data.shape()[0];
    let seq_len = mha_output.data.shape()[1];
    let d_model = mha_output.data.shape()[2];
    
    let mha_output_2d = mha_output.data.clone()
        .into_dimensionality::<ndarray::Ix3>().unwrap()
        .into_shape((batch_size * seq_len, d_model)).unwrap();
    
    let mha_output_2d_tensor = Tensor::new(mha_output_2d.into_dimensionality::<ndarray::Ix2>().unwrap());
    
    // Feed-Forward
    let feed_forward = FeedForward::new(d_model, Some(ff_dim));
    let ff_output = feed_forward.forward(&mha_output_2d_tensor);
    
    println!("Feed-Forward output shape: {:?}", ff_output.data.shape());
    
    // 9. Encoder Stack completo
    println!("\nProcessing through complete Encoder Stack...");
    use attention::EncoderStack;
    
    // Crea l'encoder stack
    let encoder_stack = EncoderStack::new(d_model, ff_dim, num_heads, num_layers, dropout_rate);
    
    // Forward pass attraverso l'encoder stack
    let encoder_output = encoder_stack.forward(&embedded_tensor, None);
    
    println!("Encoder Stack output shape: {:?}", encoder_output.data.shape());
    
    // 10. Visualizzare alcuni valori di output per conferma
    println!("\nChecking output values...");
    let encoder_data = encoder_output.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
    
    // Mostra i primi 5 valori della prima posizione
    println!("First 5 values of first position from Encoder Stack:");
    for i in 0..5.min(d_model) {
        println!("  [0, 0, {}] = {:.6}", i, encoder_data[[0, 0, i]]);
    }
    
    // 11. Creazione di una maschera causale per dimostrare il funzionamento
    println!("\nDimostrazione con maschera causale:");
    use attention::create_causal_mask;
    
    // Creiamo una maschera causale
    let causal_mask = create_causal_mask(seq_len);
    
    // Forward pass con maschera causale
    let masked_output = encoder_stack.forward(&embedded_tensor, Some(&causal_mask));
    
    println!("Encoder Stack output with causal mask shape: {:?}", masked_output.data.shape());
    
    // Verifica se ci sono differenze tra output con e senza maschera
    let masked_data = masked_output.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
    
    let mut diff_count = 0;
    let mut max_diff = 0.0;
    
    for ((b, i, j), &v1) in encoder_data.indexed_iter() {
        let v2 = masked_data[[b, i, j]];
        let diff = (v1 - v2).abs();
        if diff > 1e-5 {
            diff_count += 1;
            max_diff = f32::max(max_diff, diff);
        }
    }
    
    println!("Differenze trovate tra output con e senza maschera: {}", diff_count);
    println!("Differenza massima: {:.6}", max_diff);
    
    println!("\nPipeline completa eseguita con successo!");
    println!("La frase '{}' è stata elaborata attraverso:", input_text);
    println!("1. Tokenizzazione → {:?}", token_ids);
    println!("2. Embedding → shape {:?}", embedded.data.shape());
    println!("3. Self-Attention → shape {:?}", attention_output.data.shape());
    println!("4. Multi-Head Attention → shape {:?}", mha_output.data.shape());
    println!("5. Feed-Forward Network → shape {:?}", ff_output.data.shape());
    println!("6. Encoder Stack → shape {:?}", encoder_output.data.shape());
}

// Esempio che mostra l'uso degli embedding
fn embedding_example() {
    println!("\n--- Embedding Example ---");
    
    // 1. Crea un tokenizer semplice
    let mut tokenizer = BasicTokenizer::new();
    
    // 2. Costruisci un vocabolario minimo
    let text = "hello world transformer models are amazing for natural language processing tasks";
    tokenizer.build_vocab(text, 1);
    
    println!("Vocabolario costruito con {} token", tokenizer.get_vocab().len());
    
    // 3. Tokenizza una frase di esempio
    let example = "hello transformer models";
    let tokens = tokenizer.tokenize(example);
    let token_ids = tokenizer.encode(example);
    
    println!("Frase: '{}'", example);
    println!("Token: {:?}", tokens);
    println!("Token IDs: {:?}", token_ids);
    
    // 4. Crea un embedding
    let embedding = TransformerEmbedding::new(
        tokenizer.get_vocab().len(),  // vocab_size
        64,                           // embedding_dim
        100,                          // max_seq_len
        0.1,                          // dropout_rate
    );
    
    // 5. Passa i token attraverso l'embedding
    let embedded = embedding.forward(&token_ids);
    
    println!("Dimensione output embedding: {:?}", embedded.data.shape());
    println!("Primo token embedding generato con successo");
    
    // 6. Esempio con batch di sequenze
    let example2 = "natural language processing";
    let token_ids2 = tokenizer.encode(example2);
    
    let batch_token_ids = vec![token_ids.clone(), token_ids2];
    
    println!("\nEsempio di batch processing:");
    println!("Batch sequenze: [{}, {}]", example, example2);
    
    // Forward pass con batch
    let batch_embedded = embedding.forward_batch(&batch_token_ids);
    
    println!("Dimensione output batch embedding: {:?}", batch_embedded.data.shape());
    println!("Batch embedding generato con successo");
}

// Esempio di training di un modello semplice
fn train_simple_model() {
    println!("\n--- Training a simple model ---");
    
    // Create a model with parameters to be learned
    let x = Tensor::new(Array::from_shape_fn((1, 2), |_| 0.5));
    let mut w = Tensor::new(Array::from_shape_fn((2, 1), |_| 1.0));
    
    // Target: what we want our model to produce
    let target = Tensor::new(Array::from_elem((1, 1), 0.8));
    
    // Learning rate - use a smaller value to prevent overshooting
    let lr = 0.02;
    
    // Train for several steps
    println!("\nTraining for 20 steps:");
    for i in 0..20 {
        // Forward pass
        let out = Tensor::matmul(&x, &w);
        
        // Compute loss
        let diff_data = Array::from_shape_vec((1, 1), vec![out.data.as_slice().unwrap()[0] - target.data.as_slice().unwrap()[0]]).unwrap();
        let diff = Tensor::new(diff_data);
        let squared = Tensor::square(&diff);
        let loss = Tensor::sum(&squared);
        
        // Backward pass
        loss.backward(None);
        
        // Get gradient and update weights - unwrap safely with default
        let w_grad = w.grad.lock().unwrap().clone().unwrap_or_else(|| Array::zeros(w.data.raw_dim()));
        
        // Convert to Array2 for calculations
        let w_data = w.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w_grad_2d = w_grad.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        
        // Create new weights by subtracting gradient * learning rate
        let new_weights = &w_data - &(&w_grad_2d * lr);
        
        // Print progress
        println!("Step {}: loss = {:.6}, prediction = {:.6}, w = [{:.6}, {:.6}]", 
                i, 
                loss.data.as_slice().unwrap()[0], 
                out.data.as_slice().unwrap()[0],
                w_data[[0, 0]],
                w_data[[1, 0]]);
        
        // Re-create weight tensor with new values and reset grad
        w = Tensor::new(new_weights);
    }
    
    // Final forward pass to check result
    let final_out = Tensor::matmul(&x, &w);
    let w_data = w.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    
    println!("\nFinal prediction: {:.6} (target: 0.8)", final_out.data.as_slice().unwrap()[0]);
    println!("Final weights: w = [{:.6}, {:.6}]", w_data[[0, 0]], w_data[[1, 0]]);
    
    // For our simple network, both weights should converge to 0.8 / 2 = 0.4
    // since x = [0.5, 0.5] and we want out = 0.8
    println!("Expected optimal weights = [0.8, 0.8] for single input or [0.4, 0.4] for both inputs summing to 0.8");
}

// Esempio di training di un modello transformer
fn training_example() {
    // 1. Preparazione dei dati
    //let train_text = "Io sono un robot che impara il linguaggio naturale e posso elaborare informazioni complesse";
    let train_text = "C'era una volta una bambina di nome Stella Marie che viveva in un piccolo villaggio ai piedi di una montagna silenziosa. Stella Marie aveva occhi chiari come il cielo d'inverno e capelli scuri come la terra bagnata dopo la pioggia. Era una bambina diversa dalle altre, non perché fosse più forte o più veloce, ma perché vedeva cose che gli altri non notavano. Mentre gli altri bambini correvano nei campi e si arrampicavano sugli alberi, lei si fermava ad ascoltare i sussurri del vento tra le foglie o osservava i movimenti lenti delle lumache sotto la pioggia. Il villaggio dove viveva era circondato da una foresta che nessuno osava attraversare. Gli adulti dicevano che era pericolosa, che al suo interno vivevano spiriti antichi e animali che parlavano una lingua dimenticata dagli uomini. Ma Stella Marie non aveva paura. Ogni sera guardava quella distesa verde e sognava di scoprirne i segreti. Un giorno, mentre camminava lungo il fiume raccogliendo pietre lisce da portare a casa, vide qualcosa di strano. Un piccolo bagliore, come una lucciola, ma più intenso e costante. Lo seguì, allontanandosi sempre più dal sentiero. La luce si muoveva tra gli alberi, lenta e sicura, come se volesse guidarla. E Stella Marie la seguì. Camminò a lungo, finché il sole non cominciò a tramontare. Il cielo si fece rosa, poi arancione, poi scuro. Ma la bambina non ebbe paura. Continuava a seguire quella luce, che ora sembrava fluttuare più in alto, quasi tra i rami degli alberi. Alla fine, arrivò in una radura che non aveva mai visto prima. Al centro c'era un albero enorme, più grande di qualsiasi altro, con rami che sembravano toccare il cielo. Sotto l'albero, seduta su una radice curva, c'era una donna anziana, vestita con un mantello fatto di foglie. I suoi occhi brillavano come stelle. Benvenuta, Stella Marie disse la donna. Ti aspettavamo. La bambina non parlò. Sentiva che non servivano parole. La donna le fece cenno di avvicinarsi e le porse un piccolo seme. Questo è il seme della memoria disse. È il dono per chi vede oltre. Se lo pianterai, crescerà solo dove il cuore è puro e la verità non è dimenticata. Stella Marie prese il seme e lo tenne stretto nel palmo. Poi la donna sparì, come nebbia al sole, e la radura sembrò svanire con lei. La bambina si ritrovò di nuovo nel bosco, vicino al fiume, con il seme ancora tra le mani. Tornò a casa senza raccontare nulla. Sapeva che nessuno le avrebbe creduto. Ma da quel giorno, ogni sera, tornava in quel punto del bosco e cercava il luogo giusto dove piantare il seme. Passarono giorni, settimane, mesi. Poi, una mattina d'autunno, Stella Marie si svegliò con una strana sensazione. Uscì di casa, camminò fino al bosco e trovò un punto dove il sole filtrava tra i rami in un modo mai visto prima. Il terreno era morbido e profumava di pioggia e di muschio. Lì piantò il seme. Ogni giorno lo visitava, gli parlava, gli cantava. Un anno dopo, al suo posto, crebbe un albero diverso da tutti gli altri. I suoi frutti erano trasparenti e dentro ogni frutto si poteva vedere un ricordo: il volto di una madre, il suono di una risata, la carezza del vento su una collina lontana. Le persone del villaggio, incuriosite, cominciarono a visitare l'albero e a cogliere quei frutti. Quando li assaggiavano, ricordavano cose che avevano dimenticato, momenti felici, promesse fatte e sogni persi nel tempo. L'albero ridiede loro ciò che avevano perso. Col tempo, tutti cominciarono a rispettare Stella Marie, non come una bambina strana, ma come una custode del mistero e della memoria. Non fu mai più sola. E l'albero, che chiamarono Albero del Ricordo, divenne il cuore del villaggio. Nessuno seppe mai chi fosse davvero la donna che aveva dato a Stella Marie quel seme, né come la bambina avesse trovato il coraggio di seguire la luce nel bosco. Ma una cosa era certa: Stella Marie aveva cambiato per sempre la vita del suo villaggio. Non con la forza, non con la magia, ma con la pazienza, la fiducia e la capacità di vedere ciò che gli altri ignoravano. E ancora oggi, se si ascolta bene, nel vento tra le foglie si può sentire una voce lieve che racconta di una bambina che seguì la luce e fece crescere un albero di ricordi.";
    let test_text = "C'era una volta una bambina di nome Stella Marie che viveva in un piccolo villaggio ai piedi di una montagna silenziosa. Stella Marie aveva occhi chiari come il cielo d'inverno e capelli scuri come la terra bagnata dopo la pioggia. Era una bambina diversa dalle altre, non perché fosse più forte o più veloce, ma perché vedeva cose che gli altri non notavano. Mentre gli altri bambini correvano nei campi e si arrampicavano sugli alberi, lei si fermava ad ascoltare i sussurri del vento tra le foglie o osservava i movimenti lenti delle lumache sotto la pioggia. Il villaggio dove viveva era circondato da una foresta che nessuno osava attraversare. Gli adulti dicevano che era pericolosa, che al suo interno vivevano spiriti antichi e animali che parlavano una lingua dimenticata dagli uomini. Ma Stella Marie non aveva paura. Ogni sera guardava quella distesa verde e sognava di scoprirne i segreti. Un giorno, mentre camminava lungo il fiume raccogliendo pietre lisce da portare a casa, vide qualcosa di strano. Un piccolo bagliore, come una lucciola, ma più intenso e costante. Lo seguì, allontanandosi sempre più dal sentiero. La luce si muoveva tra gli alberi, lenta e sicura, come se volesse guidarla. E Stella Marie la seguì. Camminò a lungo, finché il sole non cominciò a tramontare. Il cielo si fece rosa, poi arancione, poi scuro. Ma la bambina non ebbe paura. Continuava a seguire quella luce, che ora sembrava fluttuare più in alto, quasi tra i rami degli alberi. Alla fine, arrivò in una radura che non aveva mai visto prima. Al centro c'era un albero enorme, più grande di qualsiasi altro, con rami che sembravano toccare il cielo. Sotto l'albero, seduta su una radice curva, c'era una donna anziana, vestita con un mantello fatto di foglie. I suoi occhi brillavano come stelle. Benvenuta, Stella Marie disse la donna. Ti aspettavamo. La bambina non parlò. Sentiva che non servivano parole. La donna le fece cenno di avvicinarsi e le porse un piccolo seme. Questo è il seme della memoria disse. È il dono per chi vede oltre. Se lo pianterai, crescerà solo dove il cuore è puro e la verità non è dimenticata. Stella Marie prese il seme e lo tenne stretto nel palmo. Poi la donna sparì, come nebbia al sole, e la radura sembrò svanire con lei. La bambina si ritrovò di nuovo nel bosco, vicino al fiume, con il seme ancora tra le mani. Tornò a casa senza raccontare nulla. Sapeva che nessuno le avrebbe creduto. Ma da quel giorno, ogni sera, tornava in quel punto del bosco e cercava il luogo giusto dove piantare il seme. Passarono giorni, settimane, mesi. Poi, una mattina d'autunno, Stella Marie si svegliò con una strana sensazione. Uscì di casa, camminò fino al bosco e trovò un punto dove il sole filtrava tra i rami in un modo mai visto prima. Il terreno era morbido e profumava di pioggia e di muschio. Lì piantò il seme. Ogni giorno lo visitava, gli parlava, gli cantava. Un anno dopo, al suo posto, crebbe un albero diverso da tutti gli altri. I suoi frutti erano trasparenti e dentro ogni frutto si poteva vedere un ricordo: il volto di una madre, il suono di una risata, la carezza del vento su una collina lontana. Le persone del villaggio, incuriosite, cominciarono a visitare l'albero e a cogliere quei frutti. Quando li assaggiavano, ricordavano cose che avevano dimenticato, momenti felici, promesse fatte e sogni persi nel tempo. L'albero ridiede loro ciò che avevano perso. Col tempo, tutti cominciarono a rispettare Stella Marie, non come una bambina strana, ma come una custode del mistero e della memoria. Non fu mai più sola. E l'albero, che chiamarono Albero del Ricordo, divenne il cuore del villaggio. Nessuno seppe mai chi fosse davvero la donna che aveva dato a Stella Marie quel seme, né come la bambina avesse trovato il coraggio di seguire la luce nel bosco. Ma una cosa era certa: Stella Marie aveva cambiato per sempre la vita del suo villaggio. Non con la forza, non con la magia, ma con la pazienza, la fiducia e la capacità di vedere ciò che gli altri ignoravano. E ancora oggi, se si ascolta bene, nel vento tra le foglie si può sentire una voce lieve che racconta di una bambina che seguì la luce e fece crescere un albero di ricordi.";
    // let test_text = "Io sono un robot che genera testo basato sul contesto fornito";

    // 2. Tokenizzazione
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab(train_text, 1);
    tokenizer.build_vocab(test_text, 1);

    println!("Vocabolario costruito con {} token", tokenizer.get_vocab().len());

    // 3. Preparazione dei dati di training
    let train_tokens = tokenizer.encode(train_text);
    let test_tokens = tokenizer.encode(test_text);

    // 4. Configurazione del modello
    let vocab_size = tokenizer.get_vocab().len();
    let d_model = 128;      // Dimensione dell'embedding
    let max_seq_len = 128; // Lunghezza massima delle sequenze
    let num_heads = 4;     // Numero di teste per multi-head attention
    let ff_dim = 128;      // Dimensione interna del feed-forward network
    let num_layers = 4;    // Numero di layer nell'encoder stack
    let dropout_rate = 0.1;
    let learning_rate = 0.001;

    // 5. Creazione del trainer, ottimizzatore e loss function
    let mut trainer = Trainer::new(
        Box::new(tokenizer.clone()) as Box<dyn Tokenizer>,
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );

    // 6. Training loop
    let num_epochs = 20;

    for epoch in 0..num_epochs {
        // Prepara gli input e i target
        // Input: tutti i token tranne l'ultimo
        // Target: tutti i token tranne il primo (predizione del token successivo)
        let input_tokens = train_tokens[0..train_tokens.len()-1].to_vec();
        let target_tokens = train_tokens[1..train_tokens.len()].to_vec();
        
        // Converti target_tokens in Array2
        let mut targets = ndarray::Array2::zeros((1, target_tokens.len()));
        for (j, &token_id) in target_tokens.iter().enumerate() {
            targets[[0, j]] = token_id;
        }
        
        // Forward pass per calcolare l'accuracy prima dell'aggiornamento
        let output = trainer.forward(&[input_tokens.clone()], Some(&targets));
        
        // Calcola l'accuracy
        let mut correct = 0;
        let total = target_tokens.len();
        
        // Estrai i logits dell'output
        let logits_data = output.logits.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        
        // Per ogni posizione, trova il token con la probabilità più alta
        for j in 0..total {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            
            for v in 0..logits_data.shape()[2] {
                let val = logits_data[[0, j, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }
            
            // Confronta con il target
            if max_idx == targets[[0, j]] as usize {
                correct += 1;
            }
        }
        
        let accuracy = (correct as f32) / (total as f32) * 100.0;
        
        // Backward pass e aggiornamento dei parametri
        let loss = trainer.train_step(
            &[input_tokens], 
            &targets
        );
        
        println!("Epoca {}/{}: loss = {:.6}, accuracy = {:.2}%", epoch + 1, num_epochs, loss, accuracy);
    }

    // 7. Generazione di testo
    let prompt = "C'era una volta";
    
    // Generazione con temperatura bassa (output più deterministico)
    let temperature_low = 0.2;
    let generated_text_low_temp = trainer.generate(
        prompt,
        15,              // max_tokens
        temperature_low, // temperatura bassa
        None             // top_k (None = usa tutti i token)
    );

    println!("Prompt: '{}'", prompt);
    println!("Testo generato (temperatura bassa): '{}'", generated_text_low_temp);

    // Generazione con temperatura alta (output più creativo)
    let temperature_high = 1.5;
    let generated_text_high_temp = trainer.generate(
        prompt,
        15,               // max_tokens
        temperature_high, // temperatura alta
        None              // top_k (None = usa tutti i token)
    );

    println!("Testo generato (temperatura alta): '{}'", generated_text_high_temp);

    // Prepara i dati di valutazione usando un closure per funzione forward
    let forward_fn = |model: &Trainer, token_ids: &Vec<Vec<usize>>, _: Option<ndarray::Array2<f32>>| {
        let mut targets = ndarray::Array2::zeros((token_ids.len(), token_ids[0].len()));
        for (i, seq) in token_ids.iter().enumerate() {
            for (j, &token_id) in seq.iter().enumerate() {
                targets[[i, j]] = token_id;
            }
        }
        model.forward(token_ids, Some(&targets))
    };

    // Prepara i dati di test
    let test_input = test_tokens[0..test_tokens.len()-1].to_vec();
    let test_dataset = vec![vec![test_input]];

    // Padding token ID (assumiamo 0 per [PAD])
    let padding_token_id = 0;

    // Valutazione
    let (avg_loss, accuracy) = training::evaluate::evaluate(
        &trainer,
        forward_fn,
        &test_dataset,
        &(Box::new(tokenizer.clone()) as Box<dyn Tokenizer>),
        Some(padding_token_id)
    );

    let perplexity = (avg_loss as f64).exp();

    println!("Loss: {:.4}", avg_loss);
    println!("Perplexity: {:.4}", perplexity);
    println!("Accuratezza: {:.2}%", accuracy * 100.0);

    // Generazione di campioni di testo
    let prompts = vec![
        "Stella Marie",
        "Ma da quel giorno",
        "E l'albero",
    ];

    training::evaluate::print_generated_samples(
        &trainer,
        forward_fn,
        &prompts.iter().map(|&s| s).collect::<Vec<&str>>(),
        &(Box::new(tokenizer.clone()) as Box<dyn Tokenizer>),
        20,   // max_new_tokens
        0.8   // temperature
    );
}

