---
markmap:
  colorFreezeLevel: 3
  initialExpandLevel: 2
---

# 🧠 Neural Network & Deep Learning Models

## ⚡ **Fundamentals**

### 🔹 The Perceptron (Building Block)
- **Formula**: `y = f(Σ wᵢxᵢ + b)`
- **Components**
  - **Weights (w)**: Learnable parameters
  - **Bias (b)**: Offset parameter
  - **Activation Function (f)**: Non-linearity
- **Learning**: Adjust weights based on error
- **Geometric Interpretation**: Linear decision boundary
- **Limitation**: Can only solve linearly separable problems

### 🔹 Activation Functions
- **ReLU**: `f(x) = max(0, x)`
  - Pros: Simple, avoids vanishing gradient
  - Cons: Dead neurons problem
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))`
  - Output: (0, 1) - good for probabilities
  - Problem: Vanishing gradient for deep networks
- **Tanh**: `f(x) = (e^x - e^(-x))/(e^x + e^(-x))`
  - Output: (-1, 1) - zero-centered
  - Better than sigmoid but still vanishing gradient
- **Leaky ReLU**: `f(x) = max(αx, x)` where α ≈ 0.01
  - Fixes dead ReLU problem
- **Swish/SiLU**: `f(x) = x × sigmoid(x)`
  - Smooth, non-monotonic
- **GELU**: Gaussian Error Linear Unit
  - Used in transformers (BERT, GPT)

### 🔹 Loss Functions
- **Mean Squared Error (MSE)**: `(1/n)Σ(y - ŷ)²`
  - For regression tasks
- **Cross-Entropy**: `-Σy log(ŷ)`
  - For classification tasks
- **Binary Cross-Entropy**: `-(y log(ŷ) + (1-y) log(1-ŷ))`
  - For binary classification
- **Huber Loss**: Combines MSE and MAE
  - Robust to outliers

### 🔹 Optimization Algorithms
- **Gradient Descent**: `w = w - η∇L`
- **SGD (Stochastic)**: Update per sample/mini-batch
- **Momentum**: `v = βv + η∇L; w = w - v`
  - Accelerates in consistent directions
- **Adam**: Adaptive learning rates
  - Combines momentum + RMSprop
  - `m = β₁m + (1-β₁)∇L; v = β₂v + (1-β₂)(∇L)²`
- **AdaGrad**: Adapts to feature frequency
- **RMSprop**: Fixes AdaGrad's decaying learning rate

## 🏗️ **Basic Architectures**

### 🔹 Feedforward Neural Networks (FNN/MLP)
- **Structure**: Input → Hidden Layer(s) → Output
- **Information Flow**: Unidirectional, no cycles
- **Universal Approximation**: Can approximate any function
- **Training**: Backpropagation algorithm
- **Hyperparameters**
  - Number of hidden layers
  - Neurons per layer
  - Learning rate
  - Batch size
- **Regularization**
  - **Dropout**: Randomly zero neurons during training
  - **Weight Decay**: L1/L2 regularization
  - **Batch Normalization**: Normalize layer inputs
- **Use Cases**: Tabular data, general classification/regression

### 🔹 Backpropagation Algorithm
- **Forward Pass**: Compute predictions layer by layer
- **Loss Computation**: Compare predictions with targets
- **Backward Pass**: Propagate gradients from output to input
- **Chain Rule**: `∂L/∂w = ∂L/∂y × ∂y/∂w`
- **Weight Update**: `w = w - η(∂L/∂w)`
- **Key Insight**: Efficiently computes gradients for all parameters

## 🖼️ **Convolutional Neural Networks (CNNs)**

### 🔹 Core Concepts
- **Purpose**: Process grid-like data (images, audio spectrograms)
- **Key Principles**
  - **Local Connectivity**: Neurons connect to local regions
  - **Weight Sharing**: Same filter across entire input
  - **Translation Invariance**: Detect features anywhere

### 🔹 CNN Layers
- **Convolutional Layer**
  - **Operation**: `(f * g)[m,n] = ΣΣ f[i,j] × g[m-i, n-j]`
  - **Filters/Kernels**: Learnable feature detectors
  - **Parameters**: Kernel size, stride, padding, channels
  - **Output Size**: `(W - F + 2P)/S + 1`
- **Pooling Layer**
  - **Max Pooling**: Take maximum in region
  - **Average Pooling**: Take average in region
  - **Purpose**: Reduce spatial dimensions, add translation invariance
- **Fully Connected Layer**: Traditional dense layer at end

### 🔹 CNN Architectures
- **LeNet-5**: Early CNN for digit recognition
- **AlexNet**: First deep CNN breakthrough (2012)
  - 8 layers, ReLU activation, dropout
- **VGGNet**: Very deep with small (3×3) filters
  - VGG-16, VGG-19 variants
- **ResNet**: Residual connections solve vanishing gradient
  - **Skip Connections**: `y = F(x) + x`
  - ResNet-50, ResNet-101, ResNet-152
- **Inception/GoogLeNet**: Multi-scale feature extraction
  - **Inception Module**: Parallel convolutions of different sizes
- **DenseNet**: Each layer connects to all previous layers
- **EfficientNet**: Optimal scaling of depth, width, resolution

### 🔹 Advanced CNN Techniques
- **1×1 Convolutions**: Channel dimension reduction
- **Depthwise Separable Convolutions**: Efficient mobile architectures
- **Dilated Convolutions**: Larger receptive fields without more parameters
- **Transpose Convolutions**: Upsampling for segmentation

## 🔄 **Recurrent Neural Networks (RNNs)**

### 🔹 Basic RNN
- **Purpose**: Process sequential data (text, time series, speech)
- **Key Feature**: Memory of previous inputs
- **Formula**: `h_t = f(W_h h_{t-1} + W_x x_t + b)`
- **Problem**: Vanishing gradient for long sequences
- **Variants**
  - **One-to-One**: Standard feedforward
  - **One-to-Many**: Image captioning
  - **Many-to-One**: Sentiment analysis
  - **Many-to-Many**: Translation, sequence labeling

### 🔹 Long Short-Term Memory (LSTM)
- **Purpose**: Solve vanishing gradient problem in RNNs
- **Key Innovation**: Cell state + gating mechanism
- **Gates**
  - **Forget Gate**: `f_t = σ(W_f[h_{t-1}, x_t] + b_f)`
  - **Input Gate**: `i_t = σ(W_i[h_{t-1}, x_t] + b_i)`
  - **Output Gate**: `o_t = σ(W_o[h_{t-1}, x_t] + b_o)`
- **Cell State Update**
  - `C̃_t = tanh(W_C[h_{t-1}, x_t] + b_C)`
  - `C_t = f_t × C_{t-1} + i_t × C̃_t`
  - `h_t = o_t × tanh(C_t)`
- **Advantages**: Handles long sequences, selective memory

### 🔹 Gated Recurrent Unit (GRU)
- **Purpose**: Simplified LSTM with fewer parameters
- **Gates**: Reset and Update gates (no separate cell state)
- **Update Gate**: `z_t = σ(W_z[h_{t-1}, x_t])`
- **Reset Gate**: `r_t = σ(W_r[h_{t-1}, x_t])`
- **Hidden State**: `h_t = (1-z_t) × h_{t-1} + z_t × tanh(W[r_t × h_{t-1}, x_t])`
- **Advantage**: Faster training, similar performance to LSTM

### 🔹 Bidirectional RNNs
- **Concept**: Process sequence in both directions
- **Architecture**: Forward RNN + Backward RNN
- **Output**: Concatenate forward and backward hidden states
- **Use Case**: When future context is available (e.g., complete sentences)

## 🎯 **Attention & Transformers**

### 🔹 Attention Mechanism
- **Problem**: RNNs bottleneck long sequences into fixed-size vector
- **Solution**: Allow model to "attend" to different parts of input
- **Types**
  - **Additive Attention**: Neural network computes alignment
  - **Multiplicative Attention**: Dot product of queries and keys
- **Advantages**: Handles long sequences, interpretable alignments

### 🔹 Self-Attention
- **Concept**: Attention within the same sequence
- **Query, Key, Value**: Different linear projections of input
- **Formula**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Multi-Head Attention**: Multiple attention heads in parallel
- **Benefits**: Capture different types of relationships

### 🔹 Transformer Architecture
- **Key Innovation**: Attention is all you need (no RNNs/CNNs)
- **Components**
  - **Encoder**: Multi-head self-attention + FFN
  - **Decoder**: Masked self-attention + encoder-decoder attention + FFN
  - **Positional Encoding**: Add position information
  - **Layer Normalization**: Stabilize training
  - **Residual Connections**: Skip connections
- **Advantages**: Parallelizable, handles long sequences, state-of-the-art results

### 🔹 Transformer Variants
- **BERT**: Bidirectional Encoder Representations
  - Pre-trained on masked language modeling
  - Fine-tuned for downstream tasks
- **GPT**: Generative Pre-trained Transformer
  - Decoder-only architecture
  - Auto-regressive text generation
- **T5**: Text-to-Text Transfer Transformer
  - All tasks as text generation
- **Vision Transformer (ViT)**: Apply transformer to image patches
- **DETR**: Object detection with transformers

## 🎨 **Generative Models**

### 🔹 Autoencoders
- **Purpose**: Learn compressed representations
- **Architecture**: Encoder → Bottleneck → Decoder
- **Loss**: Reconstruction error (input vs output)
- **Types**
  - **Vanilla Autoencoder**: Basic reconstruction
  - **Denoising Autoencoder**: Remove noise from input
  - **Sparse Autoencoder**: Encourage sparse representations
  - **Contractive Autoencoder**: Robust to input variations

### 🔹 Variational Autoencoders (VAE)
- **Purpose**: Generate new samples from learned distribution
- **Key Idea**: Learn probabilistic latent space
- **Architecture**: Encoder outputs μ and σ, decoder reconstructs from sampled z
- **Loss**: Reconstruction + KL divergence (regularization)
- **Reparameterization Trick**: `z = μ + σ × ε` (ε ~ N(0,1))
- **Applications**: Image generation, anomaly detection

### 🔹 Generative Adversarial Networks (GANs)
- **Concept**: Two networks in competition
- **Generator**: Creates fake samples from noise
- **Discriminator**: Distinguishes real from fake
- **Training**: Minimax game
  - Generator: Minimize `log(1 - D(G(z)))`
  - Discriminator: Maximize `log(D(x)) + log(1 - D(G(z)))`
- **Challenges**: Mode collapse, training instability
- **Variants**
  - **DCGAN**: Deep Convolutional GANs
  - **StyleGAN**: High-quality face generation
  - **CycleGAN**: Unpaired image-to-image translation
  - **WGAN**: Wasserstein loss for stable training

### 🔹 Diffusion Models
- **Concept**: Learn to reverse noise process
- **Forward Process**: Gradually add noise to data
- **Reverse Process**: Learn to denoise step by step
- **Training**: Predict noise added at each step
- **Advantages**: Stable training, high-quality samples
- **Examples**: DALL-E 2, Stable Diffusion, Imagen

## 📊 **Specialized Architectures**

### 🔹 Graph Neural Networks (GNNs)
- **Purpose**: Process graph-structured data
- **Key Idea**: Aggregate information from neighbors
- **Types**
  - **GCN**: Graph Convolutional Networks
  - **GraphSAGE**: Inductive learning on graphs
  - **GAT**: Graph Attention Networks
  - **GIN**: Graph Isomorphism Networks
- **Applications**: Social networks, molecular property prediction

### 🔹 Neural Architecture Search (NAS)
- **Purpose**: Automatically design neural architectures
- **Methods**: Reinforcement learning, evolutionary algorithms, differentiable
- **Examples**: EfficientNet, MobileNet families

### 🔹 Meta-Learning Networks
- **Purpose**: Learn to learn new tasks quickly
- **Approaches**
  - **Model-Agnostic Meta-Learning (MAML)**
  - **Prototypical Networks**
  - **Matching Networks**
- **Applications**: Few-shot learning, rapid adaptation

## 🚀 **Modern Large-Scale Models**

### 🔹 Large Language Models (LLMs)
- **Scale**: Billions to trillions of parameters
- **Architecture**: Transformer-based (mostly decoder-only)
- **Training**: Self-supervised on massive text corpora
- **Examples**
  - **GPT Series**: GPT-3, GPT-4, ChatGPT
  - **BERT Family**: BERT, RoBERTa, DeBERTa
  - **T5**: Text-to-text unified framework
  - **PaLM**: Pathways Language Model
  - **Claude**: Constitutional AI approach
- **Capabilities**: Text generation, reasoning, code, translation

### 🔹 Vision Foundation Models
- **Purpose**: General-purpose computer vision
- **Examples**
  - **CLIP**: Vision-language understanding
  - **SAM**: Segment Anything Model
  - **DINOv2**: Self-supervised vision features
  - **MAE**: Masked Autoencoders for vision
- **Applications**: Zero-shot classification, segmentation, detection

### 🔹 Multimodal Models
- **Purpose**: Process multiple data types simultaneously
- **Examples**
  - **GPT-4V**: Text + vision
  - **Flamingo**: Few-shot learning across modalities
  - **DALL-E**: Text to image generation
  - **Whisper**: Robust speech recognition
- **Architecture**: Shared representations across modalities

## 🔧 **Training Techniques & Optimization**

### 🔹 Advanced Training Strategies
- **Curriculum Learning**: Start with easy examples
- **Transfer Learning**: Use pre-trained models
- **Fine-tuning**: Adapt pre-trained models to new tasks
- **Multi-task Learning**: Train on multiple tasks simultaneously
- **Self-supervised Learning**: Create supervision from data itself
- **Contrastive Learning**: Learn by comparing samples

### 🔹 Regularization Techniques
- **Dropout**: Randomly zero neurons during training
- **Batch Normalization**: Normalize inputs to each layer
- **Layer Normalization**: Normalize across features (better for RNNs)
- **Weight Decay**: L1/L2 regularization on weights
- **Early Stopping**: Stop when validation performance degrades
- **Data Augmentation**: Artificially expand training data

### 🔹 Advanced Optimizers
- **AdamW**: Adam with decoupled weight decay
- **RAdam**: Rectified Adam with warm-up
- **Lookahead**: Maintain two sets of weights
- **LAMB**: Layer-wise Adaptive Moments for Batch training
- **Sharpness-Aware Minimization (SAM)**: Find flat minima

### 🔹 Learning Rate Scheduling
- **Step Decay**: Reduce LR at fixed intervals
- **Exponential Decay**: Exponentially decrease LR
- **Cosine Annealing**: Cosine-shaped LR schedule
- **Warm Restart**: Periodic LR restarts
- **One Cycle**: Single cycle of LR increase then decrease

## 🎯 **Model Selection & Evaluation**

### 🔹 Architecture Choice Guidelines
- **Tabular Data**: MLP/Feedforward networks
- **Images**: CNNs (ResNet, EfficientNet) or Vision Transformers
- **Sequential Data**: RNNs, LSTMs, or Transformers
- **Text**: Transformers (BERT for understanding, GPT for generation)
- **Graphs**: Graph Neural Networks
- **Multiple Modalities**: Multimodal transformers

### 🔹 Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression**: MSE, RMSE, MAE, R²
- **Generation**: BLEU, ROUGE (text), FID, IS (images)
- **Ranking**: NDCG, MAP, MRR

### 🔹 Model Interpretation
- **Gradient-based**: Saliency maps, integrated gradients
- **Attention Visualization**: For transformers
- **Feature Importance**: Permutation importance
- **LIME/SHAP**: Local explanations
- **Adversarial Examples**: Test model robustness

## ⚡ **Computational Considerations**

### 🔹 Efficiency Techniques
- **Model Compression**
  - **Pruning**: Remove unnecessary weights/neurons
  - **Quantization**: Use lower precision (FP16, INT8)
  - **Knowledge Distillation**: Train small model to mimic large one
- **Efficient Architectures**
  - **MobileNets**: Depthwise separable convolutions
  - **SqueezeNet**: Fire modules
  - **ShuffleNet**: Channel shuffle operations

### 🔹 Hardware Optimization
- **GPU Utilization**: Batch processing, memory management
- **Mixed Precision Training**: Use FP16 + FP32
- **Gradient Checkpointing**: Trade compute for memory
- **Model Parallelism**: Split model across devices
- **Data Parallelism**: Split data across devices