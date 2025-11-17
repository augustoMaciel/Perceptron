# Neural Network from Scratch in C

A high-performance, memory-efficient multilayer perceptron (MLP) implementation in pure C with advanced features for classification tasks.

## 🚀 Features

- **Pure C Implementation**: No external dependencies except standard libraries
- **Memory Efficient**: Optimized for large datasets with contiguous memory allocation
- **Gradient Explosion Prevention**: Advanced numerical stability mechanisms
- **Reproducible Results**: Seeded random number generator for consistent experiments
- **Adaptive Learning**: Dynamic learning rate adjustment with early stopping
- **L2 Regularization**: Prevents overfitting with configurable regularization strength
- **Progress Tracking**: Real-time training progress with sample counting
- **CSV Support**: Automatic dataset loading and preprocessing

## 🎯 Performance Benchmarks

| Dataset | Samples | Features | Classes | Architecture | Accuracy |
|---------|---------|----------|---------|--------------|----------|
| **Iris** | 150 | 4 | 3 | 2 layers (16→8) | **100.00%** |

## 🛠️ Installation

### Prerequisites
- GCC compiler
- Math library (libm)

### Build
```bash
gcc -o test test.c -lm
```

## 📖 Usage

### Interactive Mode
```bash
./test
```

### Batch Mode (Recommended)
```bash
echo "dataset.csv
num_hidden_layers
layer1_neurons
layer2_neurons
epochs
learning_rate
l2_regularization
train_percentage
random_seed" | ./test
```

### Example Configuration

#### Iris Dataset (Multi-Layer Perceptron)
```bash
echo "iris.csv
2
16
8
200
0.005
0.0001
0.75
69" | ./test
```
100% accuracy.

#### Iris Dataset (Deep Neural Network)
```bash
echo "iris.csv
3
16
32
64
3000
0.005
0.0001
0.5
42" | ./test
``
96% accuracy.

## 🔧 Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `num_hidden_layers` | Number of hidden layers | any positive integer |
| `layer_neurons` | Neurons per layer | any positive integer |
| `epochs` | Training iterations | any positive integer |
| `learning_rate` | Step size for gradient descent | any positive float |
| `l2_regularization` | Regularization strength | any positive float |
| `train_percentage` | Training data fraction | 0.01 - 1 |
| `random_seed` | Seed for reproducibility | any positive integer |

## 📊 Dataset Format

### CSV Requirements
- **Header**: No header row
- **Format**: `feature1,feature2,...,featureN,class`
- **Classes**: Integer labels starting from 0
- **Features**: Numerical values (will be normalized automatically)

### Example (Iris Dataset)
```csv
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
```

## 🧠 Architecture

### Core Components

1. **Data Management** (`dt_mn_fncs.h`)
   - Memory-efficient 2D matrix allocation
   - CSV parsing with progress tracking
   - Train/test data shuffling and splitting

2. **Neural Network** (`trn_fncs.h and nn_fncs.h`)
   - Xavier weight initialization
   - Forward propagation with sigmoid activation
   - Backpropagation with gradient clipping
   - Adaptive learning rate with patience mechanism

3. **Metrics** (`mtrcs.h`)
   - Cross-entropy loss calculation
   - Softmax output layer
   - Accuracy computation with detailed error analysis

### Key Algorithms

- **Activation Function**: Sigmoid (hidden layers), Softmax (output)
- **Loss Function**: Cross-entropy
- **Optimizer**: Gradient Descent with L2 regularization
- **Initialization**: Xavier/Glorot uniform
- **Regularization**: L2 weight decay

## 🔬 Advanced Features

### Gradient Explosion Prevention
- **NaN Detection**: Automatic training termination on numerical instability
- **Gradient Clipping**: Prevents explosive gradient growth
- **Learning Rate Adaptation**: Automatic reduction on loss plateau

### Memory Optimization
- **Contiguous Allocation**: Single malloc for entire matrices
- **Memory Reporting**: Real-time memory usage tracking
- **Efficient Storage**: Optimized data structures for any dataset size

### Reproducibility
- **Seeded Random**: Controllable randomization for experiments
- **Deterministic Training**: Same seed produces identical results
- **Scientific Rigor**: Enables proper ML methodology

## 🚨 Troubleshooting

### Common Issues

#### Gradient Explosion
**Symptoms**: Loss increases exponentially (>3.0)
**Solutions**:
- Reduce learning rate (try 0.001)
- Increase L2 regularization (try 0.001)
- Use smaller network architecture

#### Poor Convergence
**Symptoms**: Loss plateaus above 1.0
**Solutions**:
- Increase learning rate slightly
- Reduce regularization strength
- Try different architecture
- Use more training data

#### Memory Issues
**Symptoms**: Allocation failures
**Solutions**:
- Reduce training percentage
- Check available system memory
- Use smaller network architecture

### Performance Tips

1. **Start Conservative**: Use lr=0.01, L2=0.001 for small datasets
2. **Scale Architecture**: Match network size to problem complexity
3. **Monitor Training**: Watch for loss patterns and early stopping
4. **Use Seeds**: Always set random seed for reproducible experiments

## 📈 Optimization Guidelines

### Learning Rate Selection
- **Small datasets** (< 1K): 0.01 - 0.5
- **Medium datasets** (1K - 10K): 0.001 - 0.1  
- **Large datasets** (> 10K): 0.0001 - 0.01

### Architecture Design
- **Simple problems** (2-3 classes): 1-2 layers, 8-32 neurons
- **Complex problems** (4+ classes): 2-3 layers, 16-64 neurons
- **Rule of thumb**: Start small, increase if underfitting

### Regularization Strategy
- **No overfitting**: L2 = 0.0001
- **Some overfitting**: L2 = 0.001
- **Heavy overfitting**: L2 = 0.01

## 🔄 Training Process

1. **Data Loading**: Automatic CSV parsing and normalization
2. **Data Split**: Reproducible train/test split with shuffling
3. **Network Creation**: Xavier initialization of weights
4. **Training Loop**: Forward/backward propagation with progress tracking
5. **Evaluation**: Final accuracy on test set with error analysis

## 🤝 Contributing

This is a research/educational implementation. Areas for improvement:
- Additional activation functions (ReLU, tanh)
- Batch processing optimization
- Cross-validation support
- Additional optimizers (Adam, RMSprop)
- Multi-threading support

## 📝 License

This project is for educational and research purposes. Feel free to use and modify.

## 🙏 Acknowledgments

Built from scratch using fundamental machine learning principles and numerical optimization techniques.

---

**Happy Deep Learning!** 🧠✨

