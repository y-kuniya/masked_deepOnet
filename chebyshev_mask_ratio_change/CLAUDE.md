# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wave equation solving project using DeepONet (Deep Operator Networks) with Chebyshev basis functions. The project investigates mask ratio experiments comparing different input masking strategies for boundary value problems.

## Core Architecture

The project consists of four main modules:

1. **`config.py`** - Central configuration for device settings, model parameters, and training parameters
2. **`models.py`** - Shared DeepONet implementation with Branch/Trunk networks, dataset classes, and evaluation utilities
3. **`data_generator.py`** - Generates wave equation training data using Chebyshev basis functions with Dirichlet boundary conditions
4. **`mask_ratio_experiment.py`** - Runs mask ratio experiments comparing Case1 (full initial conditions) vs Case2 (masked initial conditions)
5. **`analyze_results.py`** - Analyzes and visualizes experimental results

## Data Flow

1. **Data Generation**: `data_generator.py` creates Chebyshev-based initial conditions and solves wave equations
2. **Training**: `mask_ratio_experiment.py` trains DeepONet models with different masking strategies
3. **Analysis**: `analyze_results.py` loads trained models and generates comparison plots

## Key Commands

### Data Generation
```bash
# Generate training data for a specific Chebyshev order
python data_generator.py --order 8 --samples 500 --nx 200
```

### Run Mask Ratio Experiments
```bash
# Run experiments with multiple mask ratios
python mask_ratio_experiment.py --order 5 --mask_ratios 0.4 0.3 0.2 --epochs 250
```

### Analyze Results
```bash
# Analyze and visualize results for a specific order
python analyze_results.py --order 5
```

## Directory Structure

- `data_order_{N}/` - Contains generated training data (`.npy` files)
- `result_order_{N}/` - Contains data generation visualizations
- `result_order_{N}_mask_experiment/` - Contains trained models and experiment results

## Model Configuration

The DeepONet configuration is centralized in `config.py`:
- Latent dimension: 128
- Hidden layers: 3 layers of 128 units each
- Activation: tanh
- No dropout or weight decay (for clean experiments)
- Training: 1000 epochs with batch size 1000

## Data Format

- **Branch data**: Initial conditions `(num_samples, nx)`
- **Trunk data**: Spatial coordinates `(nx,)`
- **Target data**: Final wave solutions `(num_samples, nx)`

## Key Features

- **Shared modules**: All scripts import from `models.py` and `config.py` for consistency
- **Fixed train/test splits**: Uses `random_state=42` for reproducible evaluation
- **L2 relative error**: Standard metric for comparing predictions
- **Energy conservation**: Data generation includes energy analysis for physical validation
- **Case comparison**: Case1 (full initial conditions) vs Case2 (masked initial conditions)

## Testing and Validation

No formal test framework is used. Validation is done through:
- Energy conservation checks in data generation
- L2 relative error calculations
- Visual comparison plots
- Statistical analysis of results across mask ratios