# ⚡ Baby-Hybrid AI: Implementation Guide

> **From Theory to Code**: A practical guide to building the Baby-Hybrid AI architecture for ARC-style grid reasoning tasks.

---

## 🏁 Quick Start Implementation

### Core System in 100 Lines

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class BabyHybridAI(nn.Module):
    """Minimal implementation of Baby-Hybrid AI for grid tasks"""
    
    def __init__(self, grid_size=10, embed_dim=64, num_primitives=8):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        
        # Perception: Grid → Features
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),  # 10 colors one-hot
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # Controller: Features → Program
        self.controller = nn.LSTM(embed_dim, 128, batch_first=True)
        self.program_head = nn.Linear(128, num_primitives)
        
        # Memory: Prototype storage
        self.memory = PrototypeMemory(embed_dim)
        
        # Primitives: Basic operations
        self.primitives = PrimitiveLibrary()
        
    def forward(self, grid, target=None):
        # Encode grid to features
        grid_onehot = F.one_hot(grid, 10).float().permute(0,3,1,2)
        features = self.encoder(grid_onehot)  # B×D×H×W
        
        # Retrieve similar patterns from memory
        prototypes = self.memory.retrieve(features)
        
        # Generate program
        features_flat = features.flatten(2).transpose(1,2)  # B×(H*W)×D
        lstm_out, _ = self.controller(features_flat)
        program_logits = self.program_head(lstm_out.mean(1))  # B×num_primitives
        
        # Execute program
        output_grid = self.primitives.execute(grid, program_logits)
        
        if target is not None:
            loss = F.cross_entropy(output_grid.view(-1), target.view(-1))
            return output_grid, loss
        
        return output_grid
```

---

## 🧩 Component Implementation

### 1. Advanced Encoder Architecture

```python
class PatchEncoder(nn.Module):
    """Patch-based encoder for grid perception"""
    
    def __init__(self, patch_size=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        
        # One-hot embedding layer
        self.embed = nn.Embedding(10, 16)  # 10 colors → 16D
        
        # Patch extraction and embedding
        self.patch_conv = nn.Conv2d(
            16, embed_dim, 
            kernel_size=patch_size, 
            stride=1, 
            padding=patch_size//2
        )
        
        # Attention for important patches
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, grid):
        B, H, W = grid.shape
        
        # Embed colors
        embedded = self.embed(grid)  # B×H×W×16
        embedded = embedded.permute(0, 3, 1, 2)  # B×16×H×W
        
        # Extract patch features
        patch_features = self.patch_conv(embedded)  # B×D×H×W
        
        # Apply attention across spatial dimensions
        features_flat = patch_features.flatten(2).transpose(1, 2)  # B×(H*W)×D
        attended_features, attention_weights = self.attention(
            features_flat, features_flat, features_flat
        )
        
        # Return both spatial and attended features
        return {
            'spatial': patch_features,
            'attended': attended_features,
            'attention': attention_weights,
            'contrastive': self.contrastive_head(attended_features.mean(1))
        }
```

### 2. Memory System Implementation

```python
import faiss

class PrototypeMemory:
    """Episodic memory for storing and retrieving patterns"""
    
    def __init__(self, embed_dim, max_size=10000):
        self.embed_dim = embed_dim
        self.max_size = max_size
        
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embed_dim)
        
        # Storage for associated programs/values
        self.programs = []
        self.usage_count = []
        
    def store(self, embedding, program):
        """Store a new prototype"""
        if self.index.ntotal >= self.max_size:
            self._evict_oldest()
        
        # Add to index
        embedding_np = embedding.detach().cpu().numpy().reshape(1, -1)
        self.index.add(embedding_np)
        
        # Store associated data
        self.programs.append(program)
        self.usage_count.append(0)
    
    def retrieve(self, query_embedding, k=5):
        """Retrieve k most similar prototypes"""
        if self.index.ntotal == 0:
            return []
        
        query_np = query_embedding.detach().cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_np, min(k, self.index.ntotal))
        
        # Update usage counts
        retrieved = []
        for idx in indices[0]:
            if idx >= 0:  # Valid index
                self.usage_count[idx] += 1
                retrieved.append({
                    'program': self.programs[idx],
                    'distance': distances[0][len(retrieved)]
                })
        
        return retrieved
    
    def _evict_oldest(self):
        """Simple eviction: remove least recently used"""
        min_usage_idx = np.argmin(self.usage_count)
        # Note: FAISS doesn't support direct deletion, so we'd need 
        # a more sophisticated strategy for production
        pass
```

### 3. Primitive Library

```python
class PrimitiveLibrary:
    """Collection of basic grid operations"""
    
    def __init__(self):
        self.operations = {
            'FLOOD_FILL': self.flood_fill,
            'COPY': self.copy_pattern, 
            'REPLACE': self.replace_color,
            'TRANSLATE': self.translate_pattern,
            'ROTATE': self.rotate_pattern,
            'MIRROR': self.mirror_pattern,
            'NO_OP': self.no_operation,
        }
    
    def execute(self, grid, program_logits):
        """Execute program on grid"""
        # Sample operation from logits
        operation_probs = F.softmax(program_logits, dim=-1)
        operation_idx = torch.multinomial(operation_probs, 1).item()
        
        # Map to operation name
        op_names = list(self.operations.keys())
        operation = self.operations[op_names[operation_idx]]
        
        # Execute operation (simplified - real version needs args)
        return operation(grid)
    
    def flood_fill(self, grid, start_pos=None, target_color=None):
        """Flood fill operation"""
        if start_pos is None:
            # Find a suitable starting position
            start_pos = self._find_fill_candidate(grid)
        
        if target_color is None:
            # Choose most common color as target
            target_color = torch.mode(grid.flatten()).values.item()
        
        # Implement flood fill algorithm
        result = grid.clone()
        queue = [start_pos]
        original_color = grid[start_pos].item()
        
        if original_color == target_color:
            return result
        
        while queue:
            x, y = queue.pop(0)
            if (x < 0 or x >= grid.shape[0] or 
                y < 0 or y >= grid.shape[1] or 
                result[x, y] != original_color):
                continue
            
            result[x, y] = target_color
            
            # Add neighbors
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                queue.append((x+dx, y+dy))
        
        return result
    
    def copy_pattern(self, grid, source_region=None, target_pos=None):
        """Copy a pattern to another location"""
        # Simplified implementation
        return grid  # Placeholder
    
    def replace_color(self, grid, from_color=None, to_color=None):
        """Replace all instances of one color with another"""
        if from_color is None:
            from_color = grid[0, 0].item()  # Use top-left as source
        if to_color is None:
            to_color = (from_color + 1) % 10  # Cycle colors
        
        result = grid.clone()
        result[grid == from_color] = to_color
        return result
    
    def translate_pattern(self, grid, dx=1, dy=1):
        """Translate entire pattern"""
        result = torch.zeros_like(grid)
        
        # Simple translation with bounds checking
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < grid.shape[0] and 
                    0 <= new_y < grid.shape[1]):
                    result[new_x, new_y] = grid[x, y]
        
        return result
    
    def rotate_pattern(self, grid, angle=90):
        """Rotate pattern by specified angle"""
        if angle == 90:
            return torch.rot90(grid, k=1)
        elif angle == 180:
            return torch.rot90(grid, k=2)
        elif angle == 270:
            return torch.rot90(grid, k=3)
        return grid
    
    def mirror_pattern(self, grid, axis='horizontal'):
        """Mirror pattern across axis"""
        if axis == 'horizontal':
            return torch.flip(grid, [1])
        elif axis == 'vertical':
            return torch.flip(grid, [0])
        return grid
    
    def no_operation(self, grid):
        """No-op for baseline"""
        return grid
    
    def _find_fill_candidate(self, grid):
        """Find a good position for flood fill"""
        # Simple heuristic: find edge of different colored regions
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                if (grid[i,j] != grid[i-1,j] or 
                    grid[i,j] != grid[i,j-1]):
                    return (i, j)
        return (0, 0)  # Fallback
```

---

## 🎯 Training Pipeline

### Complete Training Loop

```python
class BabyHybridTrainer:
    """Training pipeline for Baby-Hybrid AI"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Loss weights
        self.loss_weights = {
            'final': 1.0,
            'contrastive': 0.1,
            'regularization': 0.01
        }
    
    def train_phase_1_perception(self, dataloader, epochs=50):
        """Phase 1: Learn visual representations"""
        print("Phase 1: Perception Learning")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (grids, _) in enumerate(dataloader):
                grids = grids.to(self.device)
                
                # Forward pass through encoder
                encoder_output = self.model.encoder(grids)
                
                # Contrastive learning
                contrastive_loss = self._contrastive_loss(
                    encoder_output['contrastive']
                )
                
                # Backpropagation
                self.optimizer.zero_grad()
                contrastive_loss.backward()
                self.optimizer.step()
                
                total_loss += contrastive_loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Contrastive Loss = {total_loss/len(dataloader):.4f}")
    
    def train_phase_2_primitives(self, dataloader, epochs=100):
        """Phase 2: Learn primitive operations"""
        print("Phase 2: Primitive Learning")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (input_grids, target_grids) in enumerate(dataloader):
                input_grids = input_grids.to(self.device)
                target_grids = target_grids.to(self.device)
                
                # Forward pass
                output_grids, loss = self.model(input_grids, target_grids)
                
                # Calculate accuracy
                pred_flat = output_grids.view(-1)
                target_flat = target_grids.view(-1)
                correct += (pred_flat == target_flat).sum().item()
                total += target_flat.numel()
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            accuracy = 100. * correct / total
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}, "
                      f"Accuracy = {accuracy:.2f}%")
    
    def train_phase_3_composition(self, dataloader, epochs=200):
        """Phase 3: Learn program composition"""
        print("Phase 3: Program Composition")
        
        # Enable memory storage during this phase
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (input_grids, target_grids) in enumerate(dataloader):
                input_grids = input_grids.to(self.device)
                target_grids = target_grids.to(self.device)
                
                # Forward pass with memory updates
                output_grids, loss = self.model(input_grids, target_grids)
                
                # Store successful patterns in memory
                encoder_features = self.model.encoder(input_grids)
                if loss.item() < 0.1:  # Successful execution
                    self._store_in_memory(encoder_features, target_grids)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Composition Loss = {total_loss/len(dataloader):.4f}")
    
    def _contrastive_loss(self, embeddings):
        """InfoNCE contrastive loss"""
        # Simplified contrastive loss
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create positive pairs (same sample augmented)
        # In practice, you'd have augmented versions of same grid
        labels = torch.arange(batch_size).to(embeddings.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix / 0.1, labels)
        return loss
    
    def _store_in_memory(self, features, successful_output):
        """Store successful patterns in episodic memory"""
        # Extract key features for storage
        key_features = features['attended'].mean(1)  # Average across sequence
        
        # Create simple program representation
        program = self._extract_program_signature(successful_output)
        
        # Store in memory
        for i in range(key_features.shape[0]):
            self.model.memory.store(key_features[i], program[i])
    
    def _extract_program_signature(self, output_grid):
        """Extract a simple signature of the transformation"""
        # Simplified: just return the most common color as signature
        signatures = []
        for grid in output_grid:
            mode_color = torch.mode(grid.flatten()).values.item()
            signatures.append({'dominant_color': mode_color})
        return signatures
```

---

## 📊 Evaluation Framework

### Comprehensive Testing Suite

```python
class ARCEvaluator:
    """Evaluation suite for ARC-style tasks"""
    
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        
    def evaluate_comprehensive(self):
        """Run full evaluation suite"""
        results = {}
        
        # Core metrics
        results['accuracy'] = self.evaluate_accuracy()
        results['efficiency'] = self.evaluate_efficiency() 
        results['generalization'] = self.evaluate_generalization()
        results['few_shot'] = self.evaluate_few_shot()
        
        # Ablation studies
        results['ablations'] = self.run_ablations()
        
        return results
    
    def evaluate_accuracy(self):
        """Measure grid-level and cell-level accuracy"""
        self.model.eval()
        
        correct_grids = 0
        correct_cells = 0
        total_grids = 0
        total_cells = 0
        
        with torch.no_grad():
            for input_grids, target_grids in self.test_loader:
                output_grids = self.model(input_grids)
                
                # Grid-level accuracy
                perfect_matches = (output_grids == target_grids).all(dim=(1,2))
                correct_grids += perfect_matches.sum().item()
                total_grids += input_grids.shape[0]
                
                # Cell-level accuracy  
                correct_cells += (output_grids == target_grids).sum().item()
                total_cells += target_grids.numel()
        
        return {
            'grid_accuracy': correct_grids / total_grids,
            'cell_accuracy': correct_cells / total_cells
        }
    
    def evaluate_efficiency(self):
        """Measure program length and execution time"""
        # Track primitive usage during execution
        primitive_counts = []
        execution_times = []
        
        with torch.no_grad():
            for input_grids, _ in self.test_loader:
                start_time = time.time()
                
                # Execute with primitive counting
                output_grids = self.model(input_grids)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
        
        return {
            'avg_primitives': np.mean(primitive_counts) if primitive_counts else 0,
            'avg_execution_time': np.mean(execution_times)
        }
    
    def evaluate_generalization(self):
        """Test on unseen pattern types"""
        # Create test cases with novel patterns
        generalization_tasks = self._generate_novel_tasks()
        
        correct = 0
        total = len(generalization_tasks)
        
        for input_grid, target_grid in generalization_tasks:
            output_grid = self.model(input_grid.unsqueeze(0))
            if torch.equal(output_grid.squeeze(0), target_grid):
                correct += 1
        
        return correct / total
    
    def evaluate_few_shot(self, n_shots=5):
        """Test few-shot learning capability"""
        # Sample few examples for quick adaptation
        few_shot_accuracy = []
        
        for task_type in ['rotation', 'translation', 'color_swap']:
            task_examples = self._generate_task_examples(task_type, n_shots + 10)
            
            # Use first n_shots for adaptation
            adaptation_examples = task_examples[:n_shots]
            test_examples = task_examples[n_shots:]
            
            # Quick adaptation (if implemented)
            self._quick_adapt(adaptation_examples)
            
            # Test on remaining examples
            correct = 0
            for input_grid, target_grid in test_examples:
                output_grid = self.model(input_grid.unsqueeze(0))
                if torch.equal(output_grid.squeeze(0), target_grid):
                    correct += 1
            
            few_shot_accuracy.append(correct / len(test_examples))
        
        return np.mean(few_shot_accuracy)
    
    def run_ablations(self):
        """Test component importance via ablation"""
        ablation_results = {}
        
        # Test without memory
        original_memory = self.model.memory
        self.model.memory = None
        ablation_results['no_memory'] = self.evaluate_accuracy()
        self.model.memory = original_memory
        
        # Test without attention
        # ... similar ablation tests
        
        return ablation_results
    
    def _generate_novel_tasks(self):
        """Generate test cases with unseen patterns"""
        # Implementation depends on your task distribution
        return []
    
    def _generate_task_examples(self, task_type, n_examples):
        """Generate examples for specific task type"""
        # Implementation for creating synthetic tasks
        return []
    
    def _quick_adapt(self, examples):
        """Quick adaptation to new task (if meta-learning enabled)"""
        # Implementation for few-shot adaptation
        pass
```

---

## 🚀 Getting Started Checklist

### Phase 1: Minimal Prototype (Week 1)
- [ ] Set up development environment (PyTorch, CUDA)
- [ ] Implement basic grid encoder (CNN)
- [ ] Create 3 core primitives (flood_fill, replace, no_op)
- [ ] Build simple controller (direct mapping)
- [ ] Test on single-operation tasks
- [ ] Measure baseline accuracy

### Phase 2: Memory Integration (Week 2)  
- [ ] Implement prototype memory system
- [ ] Add FAISS for fast retrieval
- [ ] Integrate memory into forward pass
- [ ] Test memory-assisted learning
- [ ] Measure few-shot improvement

### Phase 3: Advanced Features (Week 3-4)
- [ ] Add contrastive learning to encoder  
- [ ] Implement world model for prediction
- [ ] Add curiosity-driven exploration
- [ ] Create comprehensive evaluation suite
- [ ] Run ablation studies

### Phase 4: Optimization (Week 5-6)
- [ ] Optimize memory usage and speed
- [ ] Implement curriculum learning
- [ ] Add more sophisticated primitives
- [ ] Test on full ARC dataset
- [ ] Compare with baselines

---

## 🔧 Development Tips

### Debugging Strategies
- **Visualize Grids**: Always plot input/output grids during development
- **Log Primitives**: Track which operations are being selected
- **Monitor Memory**: Check what patterns are being stored/retrieved
- **Gradient Flow**: Ensure gradients flow through all components
- **Incremental Testing**: Test each component in isolation first

### Performance Optimization
- **Batch Processing**: Vectorize operations across batch dimension
- **Memory Management**: Use gradient checkpointing for large models
- **Primitive Caching**: Cache results of expensive primitive operations
- **Mixed Precision**: Use FP16 training for speed gains
- **Profile Code**: Identify bottlenecks with PyTorch profiler

### Common Pitfalls
- **Memory Leaks**: Detach tensors when storing in memory
- **Device Mismatch**: Ensure all tensors on same device
- **Gradient Explosion**: Use gradient clipping
- **Overfitting**: Monitor validation performance closely
- **Primitive Bias**: Ensure balanced primitive usage

---

## 📈 Expected Results Timeline

**Week 1-2**: 
- Single operation accuracy: 60-80%
- Memory retrieval working
- Basic visualization complete

**Week 3-4**:
- Multi-operation accuracy: 40-60%  
- Contrastive learning converged
- World model predicting accurately

**Week 5-6**:
- Full system accuracy: 20-40% on simple ARC tasks
- Few-shot learning demonstrable
- Comprehensive evaluation complete

**Beyond Week 6**:
- Advanced primitive discovery
- Meta-learning integration  
- Human-level sample efficiency research