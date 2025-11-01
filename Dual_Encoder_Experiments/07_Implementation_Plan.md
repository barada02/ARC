# 💻 Implementation Plan - Technical Roadmap

## Overview

This document outlines the **technical implementation roadmap** for our dual-encoder ARC solver, including technology stack, development environment, coding standards, and detailed implementation schedules for each experimental stage.

## Technology Stack

### **Core Framework**
- **Deep Learning**: PyTorch 2.0+ (for flexibility and research features)
- **GPU Acceleration**: CUDA 11.8+ (RTX 3080/4080 class or better recommended)
- **Data Processing**: NumPy, Pandas for matrix operations and data management
- **Visualization**: Matplotlib, Seaborn for analysis plots
- **Experiment Tracking**: Weights & Biases (wandb) for experiment management

### **Development Environment**
```yaml
Environment Setup:
  Python: "3.9+"
  PyTorch: "2.0.0+"
  Dependencies:
    - torch>=2.0.0
    - torchvision>=0.15.0
    - numpy>=1.21.0
    - pandas>=1.3.0
    - matplotlib>=3.5.0
    - seaborn>=0.11.0
    - wandb>=0.13.0
    - jupyter>=1.0.0
    - tensorboard>=2.8.0
    - scikit-learn>=1.0.0
    - tqdm>=4.62.0
```

### **Project Structure**
```
arc_dual_encoder/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # ARC data loading utilities
│   │   ├── preprocessing.py    # Matrix preprocessing functions
│   │   └── augmentation.py     # Data augmentation methods
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders/
│   │   │   ├── embed_cnn.py    # Embedding + CNN encoder
│   │   │   ├── onehot_cnn.py   # One-hot + CNN encoder
│   │   │   ├── transformer.py  # Patch transformer encoder
│   │   │   └── multiscale.py   # Multi-scale encoder
│   │   ├── rule_learners/
│   │   │   ├── concat_mlp.py   # Concatenation + MLP
│   │   │   ├── cross_attention.py # Cross-attention mechanism
│   │   │   ├── contrastive.py  # Contrastive learning
│   │   │   └── hierarchical.py # Hierarchical rule learning
│   │   ├── decoders/
│   │   │   ├── cnn_decoder.py  # CNN-based decoder
│   │   │   ├── transformer_decoder.py # Transformer decoder
│   │   │   └── rule_conditioned.py # Rule-conditioned decoder
│   │   └── dual_encoder.py     # Main dual-encoder model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop implementation
│   │   ├── losses.py           # Loss function definitions
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── callbacks.py        # Training callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Model evaluation framework
│   │   ├── visualizations.py   # Result visualization
│   │   └── analysis.py         # Performance analysis tools
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logging_utils.py    # Logging utilities
│       └── matrix_utils.py     # Matrix manipulation utilities
├── experiments/
│   ├── stage1_encoders/        # Stage 1 experiments
│   ├── stage2_dual_encoder/    # Stage 2 experiments  
│   ├── stage3_rule_application/ # Stage 3 experiments
│   ├── stage4_integration/     # Stage 4 experiments
│   └── stage5_advanced/        # Stage 5 experiments
├── configs/                    # Configuration files
├── data/                       # ARC dataset (symlink to original)
├── results/                    # Experiment results and logs
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # Project documentation
```

## Stage 1: Matrix Encoding Implementation

### **Week 1-2 Implementation Schedule**

#### **Day 1-2: Project Setup and Data Loading**

**Deliverable**: Working data pipeline

```python
# src/data/loaders.py
class ARCDataLoader:
    """Load and preprocess ARC dataset"""
    
    def __init__(self, data_path, split='training'):
        self.data_path = data_path
        self.split = split
        self.tasks = self._load_tasks()
    
    def _load_tasks(self):
        """Load ARC tasks from JSON files"""
        challenges_file = f"arc-agi_{self.split}_challenges.json"
        solutions_file = f"arc-agi_{self.split}_solutions.json"
        
        with open(os.path.join(self.data_path, challenges_file)) as f:
            challenges = json.load(f)
        
        if os.path.exists(os.path.join(self.data_path, solutions_file)):
            with open(os.path.join(self.data_path, solutions_file)) as f:
                solutions = json.load(f)
        else:
            solutions = {}
        
        return self._combine_challenges_solutions(challenges, solutions)
    
    def get_task(self, task_id):
        """Get specific task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_iterator(self, batch_size=1):
        """Iterator over tasks for training"""
        task_ids = list(self.tasks.keys())
        for i in range(0, len(task_ids), batch_size):
            batch_ids = task_ids[i:i+batch_size]
            batch_tasks = [self.tasks[tid] for tid in batch_ids]
            yield batch_tasks

# Usage example
loader = ARCDataLoader("path/to/arc-prize-2025/")
for task_batch in loader.get_task_iterator(batch_size=8):
    # Process batch of tasks
    pass
```

#### **Day 3-4: Encoder Implementations**

**Deliverable**: All four encoder architectures

```python
# src/models/encoders/embed_cnn.py
class EmbedCNNEncoder(nn.Module):
    """Embedding + CNN encoder implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer for integers 0-9
        self.embedding = nn.Embedding(10, config.embed_dim)
        
        # CNN layers for spatial processing
        self.conv_layers = self._build_conv_layers()
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Linear(config.hidden_dims[-1], config.output_dim)
        
        # Decoder for reconstruction testing
        self.decoder = self._build_decoder()
    
    def _build_conv_layers(self):
        """Build convolutional layers"""
        layers = nn.ModuleList()
        in_channels = self.config.embed_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
            in_channels = hidden_dim
        
        return layers
    
    def forward(self, matrix):
        """Forward pass through encoder"""
        # matrix: (batch, height, width) with values 0-9
        batch_size, h, w = matrix.shape
        
        # Embed each cell: (batch, h, w, embed_dim)
        embedded = self.embedding(matrix)
        embedded = embedded.permute(0, 3, 1, 2)  # (batch, embed_dim, h, w)
        
        # Apply conv layers
        features = embedded
        spatial_features = []
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
            spatial_features.append(features)
        
        # Global pooling
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)
        output = self.feature_head(global_features)
        
        return {
            'global_features': output,
            'spatial_features': spatial_features,
            'final_spatial': features
        }
    
    def decode(self, encoded_features):
        """Decode features back to matrix (for testing)"""
        return self.decoder(encoded_features)
```

#### **Day 5-7: Testing and Evaluation Framework**

**Deliverable**: Comprehensive testing suite

```python
# src/evaluation/evaluator.py
class EncoderEvaluator:
    """Evaluate encoder performance"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = {}
    
    def evaluate_reconstruction(self, model, test_matrices):
        """Test reconstruction quality"""
        model.eval()
        results = []
        
        with torch.no_grad():
            for matrix, matrix_id in test_matrices:
                # Forward pass
                encoded = model(matrix.unsqueeze(0))
                reconstructed = model.decode(encoded)
                
                # Compute metrics
                perfect_match = torch.equal(reconstructed.squeeze(0), matrix)
                pixel_accuracy = (reconstructed.squeeze(0) == matrix).float().mean()
                
                results.append({
                    'matrix_id': matrix_id,
                    'perfect_reconstruction': perfect_match.item(),
                    'pixel_accuracy': pixel_accuracy.item(),
                    'feature_norm': encoded['global_features'].norm().item()
                })
        
        return results
    
    def evaluate_feature_quality(self, model, test_matrices_with_labels):
        """Evaluate learned feature representations"""
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for matrix, label in test_matrices_with_labels:
                encoded = model(matrix.unsqueeze(0))
                features.append(encoded['global_features'].cpu().numpy())
                labels.append(label)
        
        # Clustering analysis
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        features = np.vstack(features)
        unique_labels = list(set(labels))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Compute metrics
        silhouette = silhouette_score(features, cluster_labels)
        
        # Map cluster labels to true labels for ARI
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        true_labels_int = [label_to_int[label] for label in labels]
        ari = adjusted_rand_score(true_labels_int, cluster_labels)
        
        return {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari,
            'num_clusters': len(unique_labels),
            'features_shape': features.shape
        }
```

### **Stage 1 Implementation Checklist**

- [ ] **Data Loading Pipeline**: Complete ARC data loader with preprocessing
- [ ] **Encoder Implementations**: All 4 encoder architectures (EmbedCNN, OneHotCNN, Transformer, MultiScale)
- [ ] **Training Framework**: Basic training loop with loss computation
- [ ] **Evaluation Suite**: Reconstruction testing and feature analysis
- [ ] **Visualization Tools**: Matrix display and feature visualization
- [ ] **Configuration Management**: Configurable hyperparameters and model settings
- [ ] **Logging and Tracking**: Experiment logging with wandb integration
- [ ] **Unit Tests**: Basic test coverage for core functions

## Stage 2: Dual-Encoder Implementation

### **Week 3-4 Implementation Schedule**

#### **Day 1-2: Rule Learning Architectures**

**Deliverable**: Multiple rule learning approaches

```python
# src/models/rule_learners/cross_attention.py
class CrossAttentionRuleLearner(nn.Module):
    """Learn rules using cross-attention between input and output features"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Rule embedding projection
        self.rule_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.rule_dim),
            nn.ReLU(),
            nn.Linear(config.rule_dim, config.rule_dim)
        )
    
    def forward(self, input_features, output_features):
        """Learn rule from input-output feature pairs"""
        # input_features, output_features: (batch, seq_len, feature_dim)
        
        # Cross-attention: output queries attend to input keys/values
        attended_features, attention_weights = self.cross_attention(
            query=output_features.transpose(0, 1),  # (seq_len, batch, feature_dim)
            key=input_features.transpose(0, 1),
            value=input_features.transpose(0, 1)
        )
        
        # Global pooling over sequence dimension
        attended_features = attended_features.transpose(0, 1)  # (batch, seq_len, feature_dim)
        global_rule_features = attended_features.mean(dim=1)  # (batch, feature_dim)
        
        # Project to rule embedding space
        rule_embedding = self.rule_projection(global_rule_features)
        
        return {
            'rule_embedding': rule_embedding,
            'attention_weights': attention_weights,
            'attended_features': attended_features
        }
```

#### **Day 3-4: Dual-Encoder Integration**

**Deliverable**: Complete dual-encoder system

```python
# src/models/dual_encoder.py
class DualEncoderARC(nn.Module):
    """Complete dual-encoder architecture for ARC tasks"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared or separate encoders for input/output
        if config.shared_encoder:
            self.input_encoder = self._create_encoder(config.encoder_type)
            self.output_encoder = self.input_encoder
        else:
            self.input_encoder = self._create_encoder(config.encoder_type)
            self.output_encoder = self._create_encoder(config.encoder_type)
        
        # Rule learning component
        self.rule_learner = self._create_rule_learner(config.rule_learner_type)
        
        # Rule application component
        self.rule_applicator = self._create_rule_applicator(config.rule_applicator_type)
        
        # Output decoder
        self.decoder = self._create_decoder(config.decoder_type)
    
    def learn_rule_from_examples(self, examples):
        """Learn transformation rule from input-output examples"""
        rule_embeddings = []
        
        for input_matrix, output_matrix in examples:
            # Encode input and output
            input_features = self.input_encoder(input_matrix)
            output_features = self.output_encoder(output_matrix)
            
            # Learn rule for this example pair
            rule_result = self.rule_learner(input_features, output_features)
            rule_embeddings.append(rule_result['rule_embedding'])
        
        # Combine rules from multiple examples
        if len(rule_embeddings) > 1:
            # Option 1: Average rule embeddings
            combined_rule = torch.stack(rule_embeddings).mean(0)
            
            # Option 2: Attention-weighted combination
            # attention_weights = self.compute_rule_attention(rule_embeddings)
            # combined_rule = torch.sum(torch.stack(rule_embeddings) * attention_weights, dim=0)
        else:
            combined_rule = rule_embeddings[0]
        
        return combined_rule
    
    def apply_rule_to_test(self, test_input, learned_rule):
        """Apply learned rule to test input"""
        # Encode test input
        test_features = self.input_encoder(test_input)
        
        # Apply rule
        transformed_features = self.rule_applicator(test_features, learned_rule)
        
        # Decode to output matrix
        output_matrix = self.decoder(transformed_features)
        
        return output_matrix
    
    def forward(self, examples, test_input):
        """Complete forward pass: learn rule and apply to test"""
        # Learn rule from examples
        learned_rule = self.learn_rule_from_examples(examples)
        
        # Apply to test input
        predicted_output = self.apply_rule_to_test(test_input, learned_rule)
        
        return {
            'predicted_output': predicted_output,
            'learned_rule': learned_rule
        }
```

#### **Day 5-7: Training and Evaluation**

**Deliverable**: Training pipeline for dual-encoder system

```python
# src/training/trainer.py
class DualEncoderTrainer:
    """Training framework for dual-encoder models"""
    
    def __init__(self, model, config, data_loader):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.consistency_loss = self._create_consistency_loss()
        
        # Logging
        self.logger = self._setup_logging()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, task_batch in enumerate(self.data_loader.get_task_iterator()):
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            for task in task_batch:
                # Prepare examples and test case
                examples = [(ex['input'], ex['output']) for ex in task['train']]
                test_input = task['test'][0]['input']  # First test case
                test_output = task['test'][0]['output']
                
                # Convert to tensors
                examples_tensor = [(torch.tensor(inp), torch.tensor(out)) for inp, out in examples]
                test_input_tensor = torch.tensor(test_input)
                test_output_tensor = torch.tensor(test_output)
                
                # Forward pass
                result = self.model(examples_tensor, test_input_tensor)
                predicted_output = result['predicted_output']
                
                # Compute losses
                reconstruction_loss = self.criterion(
                    predicted_output.view(-1, 10),  # Flatten spatial dimensions
                    test_output_tensor.view(-1)
                )
                
                # Rule consistency loss (same rule for same task)
                consistency_loss = self.consistency_loss(result['learned_rule'], examples_tensor)
                
                # Combined loss
                task_loss = reconstruction_loss + self.config.consistency_weight * consistency_loss
                batch_loss += task_loss
            
            # Normalize by batch size
            batch_loss = batch_loss / len(task_batch)
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss.item():.4f}'
                )
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return avg_loss
```

## Stage 3-5: Advanced Implementation

### **Incremental Development Strategy**

#### **Stage 3: Rule Application Methods (Week 5-6)**
```python
# Implementation priorities:
1. Rule-conditioned decoder variants
2. Feature transformation approaches  
3. Iterative refinement mechanisms
4. Ensemble rule application methods
```

#### **Stage 4: Integration and Optimization (Week 7-8)**
```python
# Focus areas:
1. End-to-end training optimization
2. Multi-task learning framework
3. Meta-learning adaptation
4. Performance profiling and optimization
```

#### **Stage 5: Advanced Techniques (Week 9-10)**
```python
# Advanced features:
1. Attention visualization tools
2. Data augmentation pipeline
3. Architecture ensemble methods
4. Baseline comparison framework
```

## Development Standards and Practices

### **Code Quality Standards**
```python
# Type hints for all functions
def process_matrix(matrix: torch.Tensor, config: Config) -> Dict[str, torch.Tensor]:
    """Process input matrix through encoder
    
    Args:
        matrix: Input matrix of shape (batch, height, width)
        config: Configuration object with model parameters
    
    Returns:
        Dictionary containing encoded features and metadata
    """
    pass

# Comprehensive docstrings
# Error handling and validation
# Consistent naming conventions
```

### **Testing Framework**
```python
# src/tests/test_encoders.py
import pytest
import torch
from src.models.encoders import EmbedCNNEncoder

class TestEmbedCNNEncoder:
    def test_forward_pass(self):
        """Test basic forward pass"""
        config = self._get_test_config()
        model = EmbedCNNEncoder(config)
        
        # Test input
        batch_size, height, width = 2, 10, 10
        input_matrix = torch.randint(0, 10, (batch_size, height, width))
        
        # Forward pass
        output = model(input_matrix)
        
        # Assertions
        assert 'global_features' in output
        assert output['global_features'].shape == (batch_size, config.output_dim)
        assert not torch.isnan(output['global_features']).any()
    
    def test_reconstruction(self):
        """Test reconstruction accuracy on simple patterns"""
        # Implementation of reconstruction tests
        pass
```

### **Configuration Management**
```python
# configs/stage1_embed_cnn.yaml
model:
  encoder_type: "embed_cnn"
  embed_dim: 64
  hidden_dims: [128, 256, 512]
  output_dim: 1024

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  weight_decay: 0.01

data:
  data_path: "data/arc-prize-2025"
  split: "training"
  augmentation: true

logging:
  experiment_name: "stage1_embed_cnn_baseline"
  log_interval: 10
  save_interval: 50
```

## Resource Requirements and Timeline

### **Computational Resources**
- **GPU**: RTX 3080/4080 or better (12GB+ VRAM recommended)
- **RAM**: 32GB+ for larger experiments
- **Storage**: 100GB+ for datasets, models, and results
- **Time**: ~10-15 hours GPU time per major experiment

### **Development Timeline**
```gantt
title Dual-Encoder Implementation Timeline
dateFormat  YYYY-MM-DD
section Stage 1
Setup & Data Pipeline    :done, setup, 2024-11-01, 2d
Encoder Implementation   :done, encoders, after setup, 5d
Testing & Evaluation     :active, testing1, after encoders, 3d

section Stage 2  
Rule Learning Modules    :rules, after testing1, 4d
Dual-Encoder Integration :integration, after rules, 3d
Training Pipeline        :training2, after integration, 3d

section Stage 3-5
Rule Application         :application, after training2, 7d
Advanced Features        :advanced, after application, 7d
Final Integration        :final, after advanced, 7d
```

---

*This implementation plan provides a concrete technical roadmap while maintaining flexibility to adapt based on experimental findings and performance requirements.*