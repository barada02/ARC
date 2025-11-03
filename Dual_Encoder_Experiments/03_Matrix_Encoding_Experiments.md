# 🔬 Matrix Encoding Experiments - Stage 1 Detailed Plan

## Overview

Stage 1 focuses on establishing the **optimal method for encoding integer matrices (0-9) into feature representations**. This is the foundation for all subsequent experiments.

## Research Questions

### **Primary Questions**
1. **Semantic vs. Spatial**: Should we prioritize semantic understanding (color relationships) or spatial patterns?
2. **Global vs. Local**: Do we need global context or can we work with local patterns?
3. **Scalability**: How do different methods handle variable matrix sizes?
4. **Efficiency**: What's the computational cost vs. representation quality trade-off?

### **Secondary Questions**
1. How do learned features correlate with human-interpretable patterns?
2. Which approach generalizes best to unseen matrix sizes?
3. Can we identify universal features across all ARC tasks?

## Experiment 1A: Embedding + CNN Approach

### **Core Hypothesis**
Learned embeddings can capture semantic relationships between colors while CNNs preserve spatial structure.

### **Architecture Details**

```python
class EmbedCNNEncoder(nn.Module):
    def __init__(self, 
                 embed_dim=64, 
                 hidden_dims=[128, 256, 512],
                 output_dim=1024):
        super().__init__()
        
        # Color embedding: 0-9 -> embed_dim vectors
        self.embedding = nn.Embedding(10, embed_dim)
        
        # Spatial processing layers
        self.conv_layers = nn.ModuleList()
        in_channels = embed_dim
        for hidden_dim in hidden_dims:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
            in_channels = hidden_dim
            
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Linear(hidden_dims[-1], output_dim)
        
        # Reconstruction decoder (for testing)
        self.decoder = self._build_decoder(hidden_dims, embed_dim)
    
    def forward(self, matrix):
        # matrix: (batch, height, width) with values 0-9
        batch_size, h, w = matrix.shape
        
        # Embed each cell
        embedded = self.embedding(matrix)  # (batch, h, w, embed_dim)
        embedded = embedded.permute(0, 3, 1, 2)  # (batch, embed_dim, h, w)
        
        # Spatial processing
        features = embedded
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
            
        # Global representation
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)
        output = self.feature_head(global_features)
        
        return output, features  # Return both global and spatial features
```

### **Test Protocol**

#### **Dataset Generation**
```python
def generate_test_matrices():
    """Generate test matrices with known patterns"""
    matrices = []
    
    # Pattern 1: Simple color replacement
    base = np.random.randint(0, 10, (10, 10))
    transformed = np.where(base == 3, 7, base)  # Replace 3s with 7s
    matrices.append((base, transformed, "color_replacement"))
    
    # Pattern 2: Rotation
    base = create_asymmetric_pattern(8, 8)
    transformed = np.rot90(base)
    matrices.append((base, transformed, "rotation_90"))
    
    # Pattern 3: Mirroring
    base = create_asymmetric_pattern(6, 6)
    transformed = np.fliplr(base)
    matrices.append((base, transformed, "mirror_horizontal"))
    
    # Pattern 4: Scaling
    base = create_small_pattern(4, 4)
    transformed = zoom(base, 2, order=0)  # Nearest neighbor scaling
    matrices.append((base, transformed, "scaling_2x"))
    
    return matrices
```

#### **Evaluation Metrics**

```python
def evaluate_reconstruction(model, test_matrices):
    """Evaluate reconstruction quality"""
    results = {}
    
    for matrix, target, pattern_type in test_matrices:
        # Encode and reconstruct
        encoded, spatial_features = model(torch.tensor(matrix).unsqueeze(0))
        reconstructed = model.decode(encoded, spatial_features)
        
        # Measure reconstruction quality
        perfect_match = torch.equal(reconstructed, torch.tensor(matrix))
        pixel_accuracy = (reconstructed == torch.tensor(matrix)).float().mean()
        
        results[pattern_type] = {
            'perfect_reconstruction': perfect_match.item(),
            'pixel_accuracy': pixel_accuracy.item(),
            'feature_norm': encoded.norm().item()
        }
    
    return results
```

#### **Expected Results**
- **Perfect Reconstruction**: >95% on simple patterns
- **Feature Quality**: Meaningful embeddings for each color
- **Spatial Preservation**: Local patterns maintained in spatial features
- **Scalability**: Consistent performance across matrix sizes (5x5 to 30x30)

### **Ablation Studies**

#### **A1: Embedding Dimension Effects**
- Test embed_dim = [16, 32, 64, 128, 256]
- Measure reconstruction quality vs. computational cost
- **Hypothesis**: Diminishing returns after 64-128 dimensions

#### **A2: Architecture Depth**
- Test different numbers of convolutional layers [2, 4, 6, 8]
- Analyze over-fitting vs. representation power
- **Hypothesis**: 4-6 layers optimal for ARC-sized matrices

#### **A3: Spatial Resolution**
- Test with/without pooling layers
- Compare global vs. multi-scale features
- **Hypothesis**: Preserving spatial detail important for complex patterns

---

## Experiment 1B: One-Hot + CNN Approach

### **Core Hypothesis**
One-hot encoding preserves the categorical nature of colors while avoiding potential embedding biases.

### **Architecture Details**

```python
class OneHotCNNEncoder(nn.Module):
    def __init__(self, 
                 hidden_dims=[64, 128, 256, 512],
                 output_dim=1024):
        super().__init__()
        
        # No embedding - direct one-hot processing
        self.conv_layers = nn.ModuleList()
        in_channels = 10  # One-hot channels for 0-9
        
        for hidden_dim in hidden_dims:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True)
            ))
            in_channels = hidden_dim
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, matrix):
        # Convert to one-hot
        batch_size, h, w = matrix.shape
        one_hot = F.one_hot(matrix, num_classes=10).float()  # (batch, h, w, 10)
        one_hot = one_hot.permute(0, 3, 1, 2)  # (batch, 10, h, w)
        
        # Process through CNN
        features = one_hot
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
            
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)
        output = self.feature_head(global_features)
        
        return output, features
```

### **Key Comparisons with 1A**
- **Memory Usage**: One-hot uses more memory (10 channels vs embed_dim)
- **Interpretability**: Each channel represents one color explicitly
- **Learning Dynamics**: No embedding parameters to learn
- **Bias**: No learned color relationships (good or bad?)

### **Specific Tests**
1. **Color Frequency Bias**: Test on matrices with uneven color distributions
2. **Color Similarity**: Can the model learn that some colors are more similar?
3. **Efficiency**: Compare training time and memory usage with 1A

---

## Experiment 1C: Patch-Based Transformer Approach

### **Core Hypothesis**
Transformer architecture can capture long-range dependencies and complex patterns better than CNNs.

### **Architecture Details**

```python
class PatchTransformerEncoder(nn.Module):
    def __init__(self, 
                 patch_size=2,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=6,
                 output_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size**2, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))  # Max 1000 patches
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_head = nn.Linear(embed_dim, output_dim)
        
    def patchify(self, matrix):
        """Convert matrix to patches"""
        batch_size, h, w = matrix.shape
        
        # Pad if necessary
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            matrix = F.pad(matrix, (0, pad_w, 0, pad_h), value=0)
            
        h_new, w_new = matrix.shape[1], matrix.shape[2]
        
        # Extract patches
        patches = matrix.unfold(1, self.patch_size, self.patch_size)\
                       .unfold(2, self.patch_size, self.patch_size)
        # Shape: (batch, h_patches, w_patches, patch_size, patch_size)
        
        patches = patches.contiguous().view(
            batch_size, -1, self.patch_size**2
        )  # (batch, num_patches, patch_size**2)
        
        return patches
    
    def forward(self, matrix):
        # Convert to patches
        patches = self.patchify(matrix)  # (batch, num_patches, patch_size**2)
        batch_size, num_patches, _ = patches.shape
        
        # Embed patches
        embedded_patches = self.patch_embed(patches.float())
        
        # Add positional encoding
        pos_emb = self.pos_encoding[:num_patches].unsqueeze(0)
        embedded_patches += pos_emb
        
        # Transformer processing (expects seq_first=True by default)
        embedded_patches = embedded_patches.transpose(0, 1)  # (num_patches, batch, embed_dim)
        transformer_output = self.transformer(embedded_patches)
        
        # Global pooling and output
        global_features = transformer_output.mean(0)  # Average over patches
        output = self.output_head(global_features)
        
        return output, transformer_output.transpose(0, 1)
```

### **Transformer-Specific Tests**

#### **T1: Patch Size Sensitivity**
- Test patch_size = [1, 2, 3, 4]
- Measure impact on pattern recognition
- **Hypothesis**: Smaller patches better for fine details, larger for global patterns

#### **T2: Attention Pattern Analysis**
```python
def visualize_attention_patterns(model, matrix):
    """Visualize what the transformer attends to"""
    with torch.no_grad():
        # Forward pass with attention weights
        patches = model.patchify(matrix)
        embedded_patches = model.patch_embed(patches.float())
        
        # Get attention weights from each layer
        attention_weights = []
        x = embedded_patches.transpose(0, 1)
        
        for layer in model.transformer.layers:
            x, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights.append(attn_weights)
            
        return attention_weights
```

#### **T3: Long-Range Dependency Tests**
- Create matrices with patterns that span entire width/height
- Test if transformer captures these better than CNN
- **Hypothesis**: Transformer superior for global pattern recognition

---

## Experiment 1D: Multi-Scale Processing

### **Core Hypothesis**
Different ARC patterns exist at different scales - combining multi-scale processing improves representation quality.

### **Architecture Details**

```python
class MultiScaleEncoder(nn.Module):
    def __init__(self, 
                 scales=[1, 2, 4],
                 base_channels=64,
                 output_dim=1024):
        super().__init__()
        self.scales = scales
        
        # Separate encoders for each scale
        self.scale_encoders = nn.ModuleList()
        for scale in scales:
            kernel_size = 2*scale + 1  # Adaptive kernel size
            encoder = nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size, padding=scale),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels*2, kernel_size, padding=scale),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*4, kernel_size, padding=scale),
                nn.AdaptiveAvgPool2d(1)
            )
            self.scale_encoders.append(encoder)
        
        # Fusion network
        total_features = len(scales) * base_channels * 4
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, matrix):
        # Process matrix at each scale
        scale_features = []
        
        for i, (scale, encoder) in enumerate(zip(self.scales, self.scale_encoders)):
            # Convert matrix to float and add channel dimension
            x = matrix.float().unsqueeze(1)  # (batch, 1, h, w)
            
            # Optional: downsample for larger scales
            if scale > 1:
                x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                
            features = encoder(x).squeeze(-1).squeeze(-1)
            scale_features.append(features)
        
        # Fuse features from all scales
        combined_features = torch.cat(scale_features, dim=1)
        output = self.fusion(combined_features)
        
        return output, scale_features
```

### **Multi-Scale Specific Tests**

#### **M1: Scale Contribution Analysis**
```python
def analyze_scale_contributions(model, test_matrices):
    """Determine which scales contribute most to different pattern types"""
    contributions = {}
    
    for matrix, target, pattern_type in test_matrices:
        # Forward pass with individual scale analysis
        full_output, scale_features = model(matrix.unsqueeze(0))
        
        # Test each scale individually
        scale_outputs = []
        for i, scale_feature in enumerate(scale_features):
            # Reconstruct using only this scale
            single_scale_output = model.fusion(scale_feature.unsqueeze(0))
            scale_outputs.append(single_scale_output)
        
        contributions[pattern_type] = {
            f'scale_{scale}': output.norm().item() 
            for scale, output in zip(model.scales, scale_outputs)
        }
    
    return contributions
```

#### **M2: Computational Efficiency**
- Compare FLOPs and memory usage vs. single-scale approaches
- Measure inference time on different matrix sizes
- **Trade-off**: Representation quality vs. computational cost

---

## Comparative Analysis Framework

### **Cross-Experiment Comparisons**

#### **Reconstruction Quality**
```python
def compare_reconstruction_quality():
    models = [
        EmbedCNNEncoder(),
        OneHotCNNEncoder(), 
        PatchTransformerEncoder(),
        MultiScaleEncoder()
    ]
    
    test_matrices = generate_comprehensive_test_set()
    results = {}
    
    for model_name, model in zip(['EmbedCNN', 'OneHotCNN', 'Transformer', 'MultiScale'], models):
        model_results = evaluate_reconstruction(model, test_matrices)
        results[model_name] = model_results
    
    # Statistical significance testing
    for pattern_type in test_matrices[0][2]:  # pattern types
        accuracies = [results[model][pattern_type]['pixel_accuracy'] for model in results]
        # Perform ANOVA or pairwise t-tests
    
    return results
```

#### **Feature Analysis**
```python
def analyze_learned_features():
    """Compare what different models learn"""
    
    # Feature similarity analysis
    def compute_feature_similarity(model1, model2, matrices):
        similarities = []
        for matrix, _, _ in matrices:
            feat1, _ = model1(matrix.unsqueeze(0))
            feat2, _ = model2(matrix.unsqueeze(0))
            sim = F.cosine_similarity(feat1, feat2).item()
            similarities.append(sim)
        return np.mean(similarities)
    
    # Clustering analysis
    def analyze_feature_clusters(model, matrices):
        features = []
        labels = []
        for matrix, _, pattern_type in matrices:
            feat, _ = model(matrix.unsqueeze(0))
            features.append(feat.detach().numpy())
            labels.append(pattern_type)
        
        # Perform clustering and measure silhouette score
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        kmeans = KMeans(n_clusters=len(set(labels)))
        cluster_labels = kmeans.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        return score
```

### **Decision Criteria**

#### **Primary Criteria (Weight: 60%)**
1. **Reconstruction Accuracy**: Perfect reconstruction rate
2. **Pattern Generalization**: Performance on unseen pattern types
3. **Scalability**: Consistent performance across matrix sizes

#### **Secondary Criteria (Weight: 40%)**
1. **Computational Efficiency**: Training time and inference speed
2. **Feature Interpretability**: How understandable are learned features
3. **Memory Usage**: RAM and storage requirements

#### **Selection Process**
```python
def select_best_encoder(results):
    """Weighted scoring system for encoder selection"""
    
    scores = {}
    for model_name in results:
        # Primary criteria (60%)
        reconstruction_score = np.mean([r['perfect_reconstruction'] for r in results[model_name].values()])
        generalization_score = measure_generalization(model_name)
        scalability_score = measure_scalability(model_name)
        
        primary_score = 0.6 * (0.4*reconstruction_score + 0.3*generalization_score + 0.3*scalability_score)
        
        # Secondary criteria (40%)
        efficiency_score = measure_efficiency(model_name)
        interpretability_score = measure_interpretability(model_name)
        memory_score = measure_memory_usage(model_name)
        
        secondary_score = 0.4 * (0.5*efficiency_score + 0.3*interpretability_score + 0.2*memory_score)
        
        total_score = primary_score + secondary_score
        scores[model_name] = {
            'total': total_score,
            'primary': primary_score,
            'secondary': secondary_score,
            'details': {
                'reconstruction': reconstruction_score,
                'generalization': generalization_score,
                'scalability': scalability_score,
                'efficiency': efficiency_score,
                'interpretability': interpretability_score,
                'memory': memory_score
            }
        }
    
    return scores
```

## Implementation Timeline

### **Week 1: Foundation**
- **Days 1-2**: Implement all four encoder architectures
- **Days 3-4**: Create comprehensive test dataset
- **Days 5-7**: Run basic reconstruction experiments

### **Week 2: Analysis**
- **Days 1-3**: Run all ablation studies
- **Days 4-5**: Comparative analysis and feature visualization
- **Days 6-7**: Performance profiling and optimization

### **Deliverables**
1. **Code Repository**: All four encoder implementations with tests
2. **Results Report**: Comprehensive analysis of all experiments
3. **Recommendation**: Best encoder architecture with justification
4. **Documentation**: Detailed findings and lessons learned

---

*This detailed plan ensures systematic evaluation of matrix encoding approaches while providing clear criteria for selecting the optimal method for Stage 2 experiments.*