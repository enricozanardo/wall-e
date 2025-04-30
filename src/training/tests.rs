#[cfg(test)]
mod tests {
    use crate::training::{CrossEntropyLoss, AdamOptimizer, Trainer};
    use crate::nabla::tensor::Tensor;
    use crate::tokenizer::basic_tokenizer::BasicTokenizer;
    use ndarray::{Array2, Array3};
    use std::collections::HashMap;


    #[test]
    fn test_cross_entropy_loss_basic() {
        let loss_fn = CrossEntropyLoss::new();
        
        // Create simple logits: batch_size=1, seq_len=1, vocab_size=3
        let logits_data = Array3::<f32>::from_shape_vec(
            (1, 1, 3),
            vec![1.0, 2.0, 0.5]
        ).unwrap();
        
        let logits = Tensor::new_3d(logits_data);
        
        // Target: the second token (index 1) is the correct one
        let targets = Array2::<usize>::from_shape_vec(
            (1, 1),
            vec![1]
        ).unwrap();
        
        // Calculate loss and gradient
        let (loss, grad) = loss_fn.forward(&logits, &targets, None);
        
        // Verify that the loss is positive
        assert!(loss > 0.0, "The loss should be positive");
        
        // The gradient should have the same shape as the logits
        assert_eq!(grad.data.shape(), logits.data.shape());
        
        // Calculate softmax manually for comparison
        let max_logit = 2.0_f32; // The maximum value in the logits
        let exp_logits = vec![
            (1.0_f32 - max_logit).exp(),
            (2.0_f32 - max_logit).exp(),
            (0.5_f32 - max_logit).exp()
        ];
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();
        
        // The gradient of the correct token (index 1) should be (prob - 1.0)
        let grad_value = grad.data.get([0, 0, 1]).unwrap();
        assert!(
            (grad_value - (probs[1] - 1.0)).abs() < 1e-5,
            "Incorrect gradient for the correct token"
        );
        
        // The other gradients should be equal to the probabilities
        assert!(
            (grad.data.get([0, 0, 0]).unwrap() - probs[0]).abs() < 1e-5,
            "Incorrect gradient for non-target token"
        );
        assert!(
            (grad.data.get([0, 0, 2]).unwrap() - probs[2]).abs() < 1e-5,
            "Incorrect gradient for non-target token"
        );
    }

    #[test]
    fn test_cross_entropy_loss_with_padding() {
        let loss_fn = CrossEntropyLoss::new();
        
        // Create logits with padding: batch_size=2, seq_len=3, vocab_size=4
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
        
        // Target with padding (0 is the padding token)
        let targets = Array2::<usize>::from_shape_vec(
            (2, 3),
            vec![
                // Batch 0
                1, 2, 0,  // The last one is padding
                // Batch 1
                3, 1, 2,  // No padding
            ]
        ).unwrap();
        
        // Calculate loss ignoring the padding index 0
        let (loss_with_ignore, grad_with_ignore) = loss_fn.forward(&logits, &targets, Some(0));
        
        // Calculate loss without ignoring the padding index
        let (loss_without_ignore, _) = loss_fn.forward(&logits, &targets, None);
        
        // The loss without ignoring should be different
        assert!(
            (loss_with_ignore - loss_without_ignore).abs() > 1e-5,
            "The losses should be different when ignoring padding"
        );
        
        // The gradient for the padding token should be zero
        let mut sum = 0.0;
        for v in 0..4 {
            if let Some(&val) = grad_with_ignore.data.get([0, 2, v]) {
                sum += val;
            }
        }
        assert!(
            sum.abs() < 1e-5,
            "The gradient for the padding token should be zero"
        );
    }

    #[test]
    fn test_adam_optimizer_basic() {
        // Create an Adam optimizer
        let mut optimizer = AdamOptimizer::new(0.1, 0.9, 0.999, 1e-8);
        
        // Create a parameter
        let param_data = ndarray::Array2::<f32>::zeros((2, 2));
        let param = Tensor::new(param_data);
        
        // Create a unit gradient
        let grad_data = ndarray::Array2::<f32>::ones((2, 2));
        let grad = Tensor::new(grad_data);
        
        // Add parameter and gradient
        let mut params = HashMap::new();
        params.insert("w".to_string(), param);
        
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), grad);
        
        // Perform an optimization step
        optimizer.step(&mut params, &grads);
        
        // Verify that the parameters have changed
        let updated_param = params.get("w").unwrap();
        
        // The parameter should be updated in the negative direction of the gradient
        assert!(updated_param.data[[0, 0]] < 0.0, "The parameter should decrease");
        
        // The value should be about -0.1 (learning rate)
        // But with Adam correction factors
        assert!(
            (updated_param.data[[0, 0]] + 0.1).abs() < 0.01,
            "The update is not of the expected size"
        );
    }

    #[test]
    fn test_adam_optimizer_momentum() {
        // Create an Adam optimizer
        let mut optimizer = AdamOptimizer::new(0.1, 0.9, 0.999, 1e-8);
        
        // Create a parameter
        let param_data = ndarray::Array2::<f32>::zeros((1, 1));
        let param = Tensor::new(param_data);
        
        // Add parameter
        let mut params = HashMap::new();
        params.insert("w".to_string(), param);
        
        // First iteration: positive gradient
        let grad_pos = Tensor::new(ndarray::Array2::<f32>::from_elem((1, 1), 1.0));
        let mut grads_pos = HashMap::new();
        grads_pos.insert("w".to_string(), grad_pos);
        optimizer.step(&mut params, &grads_pos);
        
        let _after_pos = params.get("w").unwrap().data[[0, 0]];
        
        // Second iteration: negative gradient
        let grad_neg = Tensor::new(ndarray::Array2::<f32>::from_elem((1, 1), -1.0));
        let mut grads_neg = HashMap::new();
        grads_neg.insert("w".to_string(), grad_neg);
        optimizer.step(&mut params, &grads_neg);
        
        let after_neg = params.get("w").unwrap().data[[0, 0]];
        
        // The momentum should dampen the effect of the sign change
        assert!(
            after_neg < 0.0,
            "The momentum effect should maintain the negative direction"
        );
    }

    #[test]
    fn test_trainer_initialization() {
        // Create a basic tokenizer
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.build_vocab("Questo è un test di un trainer model", 1);
        
        // Model parameters
        let model_dim = 8;
        let ff_dim = 16;
        let num_heads = 2;
        let num_layers = 1;
        let dropout_rate = 0.1;
        let learning_rate = 0.001;
        
        // Create the trainer
        let trainer = Trainer::new(
            Box::new(tokenizer),
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
            learning_rate,
        );
        
        // Verify that the structure has been initialized correctly
        assert_eq!(trainer.model_dim, model_dim);
        
        // Verify that the output projection has been initialized
        assert!(trainer.params.contains_key("output_projection"));
    }

    #[test]
    fn test_trainer_forward_shape() {
        // Create a minimal tokenizer
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.build_vocab("a b c", 1);
        
        let vocab_size = 5; // Special tokens ([PAD], [UNK]) + (a, b, c)
        
        // Model parameters
        let model_dim = 8;
        let ff_dim = 16;
        let num_heads = 2;
        let num_layers = 1;
        let dropout_rate = 0.0; // Disable dropout for tests
        let learning_rate = 0.001;
        
        // Create the trainer
        let trainer = Trainer::new(
            Box::new(tokenizer),
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
            learning_rate,
        );
        
        // Create a batch with one sequence
        let batch = vec![vec![1, 2, 3]];
        
        // Forward pass
        let output = trainer.forward(&batch, None);
        
        // Verify the dimensions of the output
        let shape = output.logits.data.shape();
        assert_eq!(shape.len(), 3, "The output should be a 3D tensor");
        assert_eq!(shape[0], 1, "Incorrect batch size");
        assert_eq!(shape[1], 3, "Incorrect sequence length");
        assert_eq!(shape[2], vocab_size, "Incorrect vocabulary size");
    }

    #[test]
    fn test_trainer_train_step() {
        // Create a minimal tokenizer
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.build_vocab("a b c", 1);
        
        // Model parameters
        let model_dim = 8;
        let ff_dim = 16;
        let num_heads = 2;
        let num_layers = 1;
        let dropout_rate = 0.0; // Disable dropout for tests
        let learning_rate = 0.01; // Higher learning rate for testing
        
        // Create the trainer
        let mut trainer = Trainer::new(
            Box::new(tokenizer),
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
            learning_rate,
        );
        
        // Save the original parameters
        let original_output_proj = trainer.output_projection.data.clone();
        
        // Create a batch with one sequence
        let batch = vec![vec![1, 2]];
        
        // Create target
        let targets = Array2::<usize>::from_shape_vec(
            (1, 2),
            vec![2, 3]
        ).unwrap();
        
        // Perform a training step
        let loss = trainer.train_step(&batch, &targets);
        
        // Verify that the loss is positive
        assert!(loss > 0.0, "The loss should be positive");
        
        // Verify that the parameters have changed
        assert!(
            (trainer.output_projection.data.clone() - original_output_proj)
                .mapv(|x| x.abs())
                .sum() > 0.0,
            "The parameters should be updated after train_step"
        );
    } 

}