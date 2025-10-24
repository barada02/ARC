---
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 1
---

# 🌐 Artificial Intelligence (AI)

## 1. Major Branches

### 🤖 Machine Learning (ML)
- Algorithms that learn patterns from data
- Two core families:
  - **Classical ML (Non-Neural)**
  - **Deep Learning (Neural Network–Based)**

## 2. 🧩 Classical / Non-Neural Machine Learning
### 🧮 Core Idea
- Each model is a **distinct mathematical module**
- Learns via **explicit formulas**, **probabilistic methods**, or **optimization**
- No concept of neurons or layers

### 📊 Common Models
- **Linear Regression**
  - Predicts continuous values
  - Minimizes Mean Squared Error
- **Logistic Regression**
  - Classifies with probability (sigmoid)
  - Uses maximum likelihood estimation
- **SVM (Support Vector Machine)**
  - Finds max-margin hyperplane
  - Uses convex optimization
- **Decision Tree**
  - Splits data using impurity (Gini/entropy)
  - Produces if–else rules
- **Random Forest**
  - Ensemble of many trees
  - Averaging/voting improves accuracy
- **KNN (K-Nearest Neighbors)**
  - Memorizes data
  - Predicts based on nearby samples
- **Naive Bayes**
  - Uses Bayes’ theorem with independence assumption
- **PCA (Principal Component Analysis)**
  - Finds directions of maximum variance
  - Uses linear algebra (eigenvectors)

### ⚙️ Learning Mechanism
- Optimizes **explicit objective functions**
- Examples:
  - Linear Regression → minimize squared error
  - SVM → maximize class margin
  - Decision Tree → minimize impurity
- Uses **analytical or geometric optimization**, not backpropagation

## 3. 🧠 Neural Network / Deep Learning
### 💡 Foundational Unit: The Perceptron
- Mathematical formula: `y = f(Σ wᵢxᵢ + b)`
- `f()` = activation (ReLU, sigmoid, tanh, etc.)
- Learns through **weight updates**

### 🔄 Learning Mechanism
- **Gradient Descent + Backpropagation**
  - Forward pass → compute output
  - Compare to true output (loss)
  - Backward pass → update weights via gradients
- Objective: minimize prediction error

### 🧱 Building Structure
- **Layers**
  - Input → Hidden → Output
- **Networks**
  - Stack multiple layers → feedforward networks
- **Activation Functions**
  - Add nonlinearity → allow complex mapping

### 🏗️ Major Neural Architectures
- **Feedforward NN (FNN / MLP)**
  - Simple multilayer network
- **CNN (Convolutional Neural Network)**
  - Local connections, weight sharing
  - Best for images
- **RNN (Recurrent Neural Network)**
  - Temporal memory
  - Best for sequences
- **LSTM / GRU**
  - Improved RNNs for long-term memory
- **Transformer**
  - Uses self-attention instead of recurrence
  - Foundation for GPT, BERT, etc.
- **Autoencoder**
  - Learns compressed representations
- **GAN (Generative Adversarial Network)**
  - Generator + Discriminator
- **Diffusion Models**
  - Denoising-based generation (DALL·E, Stable Diffusion)
- **GNN (Graph Neural Network)**
  - Works on graph-structured data

## 4. 🧬 Scaling Up → Foundation Models
- **LLMs (Large Language Models)**
  - Massive Transformer networks (GPT, Claude, Gemini)
- **Vision Foundation Models**
  - CNN/ViT-based (SAM, CLIP, DINOv2)
- **Multimodal Models**
  - Combine text, image, audio (GPT-4, Gemini 1.5)

## 5. 🧰 AI Systems and Applications
- Chatbots (ChatGPT, Claude)
- Image generators (DALL·E, Midjourney)
- Speech systems (Whisper)
- Recommendation engines
- Autonomous agents (AutoGPT, Devin)
- Robotics and control systems

## 6. ⚖️ Neural vs. Non-Neural Summary
| Feature | Neural Network Models | Classical ML Models |
|----------|----------------------|---------------------|
| **Base Unit** | Perceptron (universal) | Distinct mathematical model |
| **Learning** | Gradient descent, backprop | Explicit optimization or logic |
| **Data Need** | High | Low–medium |
| **Interpretability** | Low (black box) | High (transparent math) |
| **Computation** | Heavy (GPU) | Light (CPU) |
| **Power** | High for complex patterns | Limited for simple ones |

## 7. 🧭 Big Picture Hierarchy
