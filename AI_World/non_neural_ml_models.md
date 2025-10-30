---
markmap:
  colorFreezeLevel: 3
  initialExpandLevel: 2
---

# 🧩 Non-Neural Machine Learning

## 🤖 **ML Models**

### 📈 **Linear Models**

#### 🔹 Linear Regression
- **Purpose**: Predict continuous values
- **Formula**: `y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`
- **Loss Function**: Mean Squared Error (MSE)
  - `MSE = (1/n) Σ(y_actual - y_predicted)²`
- **Optimization**: Analytical solution or Gradient Descent
- **Assumptions**
  - Linear relationship between features and target
  - Independence of errors
  - Homoscedasticity (constant variance)
  - Normal distribution of errors
- **Pros**: Simple, interpretable, fast training
- **Cons**: Assumes linearity, sensitive to outliers
- **Use Cases**: Price prediction, sales forecasting

#### 🔹 Logistic Regression
- **Purpose**: Binary/multiclass classification
- **Formula**: `p = 1/(1 + e^(-z))` where `z = w₀ + w₁x₁ + ...`
- **Loss Function**: Log-likelihood/Cross-entropy
- **Decision Boundary**: Linear in feature space
- **Outputs**: Probabilities (0-1)
- **Optimization**: Maximum Likelihood Estimation
- **Regularization**
  - L1 (Lasso): Feature selection
  - L2 (Ridge): Prevents overfitting
- **Pros**: Probabilistic output, no tuning needed
- **Cons**: Assumes linear decision boundary
- **Use Cases**: Email spam detection, medical diagnosis

#### 🔹 Polynomial Regression
- **Purpose**: Capture non-linear relationships
- **Method**: Add polynomial features (x², x³, xy, etc.)
- **Risk**: High degree → overfitting
- **Regularization**: Essential for higher degrees

### 🌳 **Tree-Based Models**

#### 🔹 Decision Trees
- **Purpose**: Classification and regression
- **Structure**: Binary splits creating if-else rules
- **Splitting Criteria**
  - **Gini Impurity**: `1 - Σpᵢ²` (classification)
  - **Entropy**: `-Σpᵢlog₂(pᵢ)` (classification)
  - **MSE**: Mean squared error (regression)
- **Algorithm**: Greedy recursive splitting
- **Hyperparameters**
  - `max_depth`: Tree depth limit
  - `min_samples_split`: Min samples to split
  - `min_samples_leaf`: Min samples in leaf
- **Pros**: Interpretable, handles mixed data types
- **Cons**: Prone to overfitting, unstable
- **Pruning**: Remove branches to reduce overfitting

#### 🔹 Random Forest
- **Purpose**: Ensemble of decision trees
- **Method**: Bootstrap Aggregating (Bagging)
- **Process**
  1. Create multiple bootstrap samples
  2. Train tree on each sample
  3. Random subset of features at each split
  4. Average predictions (regression) or vote (classification)
- **Hyperparameters**
  - `n_estimators`: Number of trees
  - `max_features`: Features per split
  - `bootstrap`: Sampling method
- **Feature Importance**: Based on impurity reduction
- **Pros**: Reduces overfitting, handles missing values
- **Cons**: Less interpretable, can overfit with noise
- **OOB Score**: Out-of-bag validation estimate

#### 🔹 Gradient Boosting Trees
- **Purpose**: Sequential ensemble learning
- **Method**: Each tree corrects previous errors
- **Popular Implementations**
  - **XGBoost**: Extreme Gradient Boosting
  - **LightGBM**: Light Gradient Boosting
  - **CatBoost**: Categorical Boosting
- **Key Concepts**
  - Learning rate: Controls step size
  - Regularization: L1, L2 penalties
  - Early stopping: Prevent overfitting
- **Pros**: High accuracy, handles mixed data
- **Cons**: Prone to overfitting, requires tuning

### 📏 **Distance-Based Models**

#### 🔹 K-Nearest Neighbors (KNN)
- **Purpose**: Classification and regression
- **Method**: "Lazy learning" - stores all training data
- **Prediction Process**
  1. Calculate distance to all training points
  2. Find K nearest neighbors
  3. Classification: Majority vote
  4. Regression: Average of K values
- **Distance Metrics**
  - **Euclidean**: `√Σ(xᵢ - yᵢ)²`
  - **Manhattan**: `Σ|xᵢ - yᵢ|`
  - **Hamming**: For categorical data
- **Hyperparameters**
  - `k`: Number of neighbors
  - `weights`: uniform vs distance-based
- **Pros**: Simple, works with non-linear data
- **Cons**: Computationally expensive, sensitive to scale
- **Preprocessing**: Feature scaling essential

#### 🔹 K-Means Clustering
- **Purpose**: Unsupervised clustering
- **Algorithm**
  1. Initialize K cluster centers
  2. Assign points to nearest center
  3. Update centers to mean of assigned points
  4. Repeat until convergence
- **Objective**: Minimize within-cluster sum of squares
- **Initialization**: K-means++, random
- **Determining K**: Elbow method, silhouette analysis
- **Pros**: Simple, efficient for spherical clusters
- **Cons**: Assumes spherical clusters, needs K

### 🎯 **Support Vector Machines (SVM)**

#### 🔹 Linear SVM
- **Purpose**: Classification with maximum margin
- **Concept**: Find hyperplane with largest margin
- **Support Vectors**: Points closest to decision boundary
- **Objective**: Maximize margin between classes
- **Soft Margin**: Allow some misclassification
- **C Parameter**: Regularization strength
  - High C: Hard margin (low bias, high variance)
  - Low C: Soft margin (high bias, low variance)

#### 🔹 Non-Linear SVM (Kernel Trick)
- **Purpose**: Handle non-linearly separable data
- **Method**: Map data to higher dimension
- **Common Kernels**
  - **RBF (Gaussian)**: `K(x,y) = exp(-γ||x-y||²)`
  - **Polynomial**: `K(x,y) = (γ⟨x,y⟩ + r)^d`
  - **Sigmoid**: `K(x,y) = tanh(γ⟨x,y⟩ + r)`
- **Hyperparameters**
  - `C`: Regularization
  - `γ` (gamma): Kernel coefficient
- **Pros**: Effective in high dimensions, memory efficient
- **Cons**: Slow on large datasets, no probability estimates

### 📊 **Probabilistic Models**

#### 🔹 Naive Bayes
- **Purpose**: Classification using Bayes' theorem
- **Formula**: `P(class|features) = P(features|class) × P(class) / P(features)`
- **"Naive" Assumption**: Features are independent
- **Variants**
  - **Gaussian NB**: Continuous features (assumes normal distribution)
  - **Multinomial NB**: Count data (text classification)
  - **Bernoulli NB**: Binary features
- **Pros**: Fast, works with small data, handles multiple classes
- **Cons**: Independence assumption rarely true
- **Use Cases**: Text classification, spam filtering

#### 🔹 Gaussian Mixture Models (GMM)
- **Purpose**: Soft clustering, density estimation
- **Assumption**: Data comes from mixture of Gaussian distributions
- **Algorithm**: Expectation-Maximization (EM)
- **Output**: Probability of belonging to each cluster
- **Parameters**: Means, covariances, mixing coefficients
- **Model Selection**: BIC, AIC for number of components

## 🔧 **ML Techniques & Best Practices**

### 🔍 **Dimensionality Reduction**

#### 🔹 Principal Component Analysis (PCA)
- **Purpose**: Reduce dimensionality while preserving variance
- **Method**: Find directions of maximum variance
- **Process**
  1. Standardize data
  2. Compute covariance matrix
  3. Find eigenvalues and eigenvectors
  4. Select top k components
- **Explained Variance Ratio**: How much variance each PC captures
- **Pros**: Reduces overfitting, visualization
- **Cons**: Linear transformation, less interpretable
- **Applications**: Preprocessing, visualization, compression

#### 🔹 Linear Discriminant Analysis (LDA)
- **Purpose**: Dimensionality reduction for classification
- **Goal**: Maximize between-class variance, minimize within-class variance
- **Difference from PCA**: Supervised (uses class labels)
- **Output**: Linear discriminants for classification

#### 🔹 t-SNE
- **Purpose**: Non-linear dimensionality reduction for visualization
- **Method**: Preserve local neighborhood structure
- **Hyperparameters**
  - `perplexity`: Local neighborhood size
  - `learning_rate`: Step size
- **Pros**: Great for visualization, captures non-linear structure
- **Cons**: Computationally expensive, not deterministic

### 🎲 **Ensemble Methods**

#### 🔹 Bagging
- **Purpose**: Reduce variance through averaging
- **Method**: Train multiple models on bootstrap samples
- **Examples**: Random Forest, Extra Trees
- **Benefits**: Reduces overfitting, parallel training

#### 🔹 Boosting
- **Purpose**: Reduce bias through sequential learning
- **Method**: Each model corrects previous errors
- **Examples**: AdaBoost, Gradient Boosting, XGBoost
- **Benefits**: High accuracy, handles weak learners

#### 🔹 Voting Classifiers
- **Hard Voting**: Majority class prediction
- **Soft Voting**: Average predicted probabilities
- **Requirement**: Diverse base models for best results

### 🔧 **Model Evaluation & Selection**

#### 🔹 Cross-Validation
- **K-Fold CV**: Split data into k folds
- **Stratified CV**: Maintains class distribution
- **Leave-One-Out CV**: K = n (number of samples)
- **Time Series CV**: Respect temporal order

#### 🔹 Metrics
- **Classification**
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC, Confusion Matrix
- **Regression**
  - MSE, RMSE, MAE, R²
- **Clustering**
  - Silhouette Score, Adjusted Rand Index

#### 🔹 Hyperparameter Tuning
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameters
- **Bayesian Optimization**: Smart parameter search
- **Validation Strategy**: Always use separate validation set

### 🎯 **Algorithm Selection Guide**

#### 🔹 By Problem Type
- **Linear Separable**: Logistic Regression, Linear SVM
- **Non-Linear**: Kernel SVM, Tree-based models
- **High Dimensions**: Naive Bayes, Linear models
- **Small Dataset**: Naive Bayes, KNN
- **Large Dataset**: Linear models, Gradient Boosting
- **Interpretability**: Decision Trees, Linear Regression
- **Speed**: Naive Bayes, Linear models

#### 🔹 By Data Characteristics
- **Mixed Data Types**: Tree-based models
- **Text Data**: Naive Bayes, Linear models
- **Outliers Present**: Tree-based models, Robust regression
- **Missing Values**: Tree-based models (some implementations)
- **Categorical Features**: Tree-based models, Naive Bayes