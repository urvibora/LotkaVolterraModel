# 
# Universal Differential Equations for Lotka-Volterra System

This project demonstrates the use of Universal Differential Equations (UDEs) to learn unknown interaction terms in the classic Lotka-Volterra predator-prey model using Julia's scientific machine learning ecosystem.

## Overview

The Lotka-Volterra system is a classic model in mathematical biology that describes the dynamics of two species - predators and prey. This implementation uses UDEs to learn the nonlinear interaction terms (-βxy and +γxy) while keeping the linear terms fixed, showcasing how neural networks can be embedded within differential equations to discover unknown physics.

### Mathematical Background

The true Lotka-Volterra system is:
```
dx/dt = αx - βxy     (prey equation)
dy/dt = -δy + γxy    (predator equation)
```

Our UDE replaces the interaction terms with neural networks:
```
dx/dt = αx + NN₁(x,y)     where NN₁ learns -βxy
dy/dt = -δy + NN₂(x,y)    where NN₂ learns +γxy
```

## Features

- **Synthetic Data Generation**: Creates ground truth Lotka-Volterra dynamics
- **Universal Differential Equations**: Hybrid model combining known physics with neural networks
- **Two-Stage Training**: ADAM optimizer followed by BFGS for fine-tuning
- **Comprehensive Visualization**: Training progress, learned interactions, and model performance
- **Partial Data Training**: Trains on only 30% of data to test extrapolation

## Requirements

```julia
using Lux, DiffEqFlux, DifferentialEquations, Random, ComponentArrays, LinearAlgebra
using Optimization, OptimizationOptimJL, OptimizationOptimisers, Statistics
using Plots
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ude-lotka-volterra.git
cd ude-lotka-volterra
```

2. Install Julia dependencies:
```julia
using Pkg
Pkg.add(["Lux", "DiffEqFlux", "DifferentialEquations", "Random", "ComponentArrays", 
         "LinearAlgebra", "Optimization", "OptimizationOptimJL", "OptimizationOptimisers", 
         "Statistics", "Plots"])
```

## Usage

Simply run the main script:
```julia
julia lotka_volterra_ude.jl
```

The script will:
1. Generate synthetic Lotka-Volterra data
2. Train the UDE model on the first 30% of data
3. Generate visualizations comparing true vs learned dynamics
4. Save plots to PNG files

## Key Parameters

- **System Parameters**: α=1.5, β=1.0, γ=1.0, δ=3.0
- **Initial Conditions**: x₀=1.0, y₀=1.0 (prey and predator populations)
- **Time Span**: 0 to 10 time units
- **Training Fraction**: 30% of total data
- **Network Architecture**: Two separate networks with 10 hidden units each

## Output

The script generates several visualizations:

1. **`true_dynamics.png`**: Complete Lotka-Volterra dynamics with training/testing split
2. **`training_results.png`**: Comparison of true data vs UDE predictions on training set
3. **`interaction_comparison.png`**: True vs learned interaction terms (-βxy and +γxy)
4. **`TrainingLoss.png`**: Training loss progression through ADAM and BFGS phases

## Results

The UDE successfully learns to approximate the nonlinear interaction terms, demonstrating:
- Accurate reconstruction of predator-prey oscillations
- Discovery of interaction terms without prior knowledge of their functional form
- Robust training with two-stage optimization strategy

## Technical Details

### Neural Network Architecture
- Two separate networks for each interaction term
- Input: [x, y] (current populations)
- Hidden layer: 10 neurons with tanh activation
- Output: Single value representing interaction strength

### Training Strategy
1. **ADAM Phase**: 5000 iterations with learning rate 0.01
2. **BFGS Phase**: Up to 1000 iterations for fine-tuning
3. **Automatic Differentiation**: Zygote.jl for gradient computation
4. **Adjoint Sensitivity**: InterpolatingAdjoint for efficient gradient calculation

### Loss Function
Mean squared error between predicted and true trajectories on training data.

## Extensions

This framework can be extended to:
- Learn more complex interaction terms
- Handle noisy observations
- Incorporate uncertainty quantification
- Apply to other dynamical systems in biology, physics, or engineering

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Performance improvements
- Additional visualization options
