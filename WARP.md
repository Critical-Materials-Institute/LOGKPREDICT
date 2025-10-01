# WARP.md

This file provides guidance when working with code in this repository.

## Project Overview

LOGKPREDICT is a computational chemistry tool that predicts stability constants (log K values) for metal-ligand complexes. It integrates with Chemprop machine learning models and uses RDKit for molecular processing. The codebase maintains dual compatibility: a legacy executable interface for HostDesigner integration and a modern Python API.

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -n logkpredict python=3.9 -y
conda activate logkpredict
conda install -c conda-forge rdkit numpy pandas -y

# Install Chemprop and dependencies
pip install chemprop>=1.6.1 click typing-extensions

# Install in development mode
pip install -e .

# Install development tools
pip install pytest black isort flake8 mypy

# Set required environment variable
export LOGKPREDICT_DIR="${PWD}"
```

### Code Quality

```bash
# Format code
black logk_lib/
isort logk_lib/

# Lint code
flake8 logk_lib/

# Type checking
mypy logk_lib/

# Run all checks at once
black logk_lib/ && isort logk_lib/ && flake8 logk_lib/ && mypy logk_lib/
```

### Testing

```bash
# Test original executable interface
./LOGKPREDICT

# Test Python API
python -c "
from logk_lib import LogKPredictor
predictor = LogKPredictor()
result = predictor.predict_from_file('logk_input')
print(f'Predicted log K: {result}')
"

# Run unit tests (if/when implemented)
pytest tests/
```

### Building and Installing

```bash
# Build package
python -m build

# Install locally
pip install -e .

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Code Architecture

### Dual Interface Design

The codebase maintains two interfaces:

1. **Legacy executable**: `LOGKPREDICT` - Monolithic script for HostDesigner compatibility
2. **Modern Python API**: `logk_lib/` - Modular library with clean separation of concerns

### Core Components

**`logk_lib/predictor.py`**: Main prediction orchestrator
- Coordinates molecular processing and ML prediction
- Manages Chemprop subprocess execution
- Handles feature combination and masking for model compatibility
- File I/O for input parsing and output generation

**`logk_lib/molecular_processing.py`**: RDKit molecular operations
- Calculates specific molecular descriptors (indices 100-140 from RDKit's 200 descriptors)
- Converts metal-ligand bonds to dative bonds for proper complex representation
- Generates SMILES with cleaned dative bond notation
- Identifies transition metals vs. common non-transition elements

**`logk_lib/exceptions.py`**: Comprehensive error hierarchy
- `LogKPredictError` base class with specific subclasses
- Enables precise error handling and debugging

### Critical Implementation Details

**Feature Masking**: The model uses a specific feature mask (`FEATURE_MASK`) that selects only certain molecular descriptors and input features. This mask is hardcoded and critical for model compatibility.

**Dative Bond Handling**: Metal complexes require special bond representation where N/O atoms form dative bonds with transition metals. The molecular processing pipeline converts standard bonds to dative bonds and cleans the resulting SMILES representation.

**Chemprop Integration**: Predictions are made by creating temporary CSV files and calling the `chemprop_predict` command-line tool, then parsing the results.

### File Processing Flow

1. Parse `logk_input` file containing features and MOL block
2. Extract numerical features (starting from index 2 or 3 depending on version)
3. Convert MOL block to RDKit molecule with dative bonds
4. Calculate molecular descriptors (specific RDKit subset)
5. Combine features with descriptors and apply feature mask
6. Generate SMILES and create Chemprop input files
7. Execute Chemprop prediction via subprocess
8. Parse prediction result and write to `logk_output`

### Key Constants and Configuration

- **Environment Variable**: `LOGKPREDICT_DIR` must be set to the directory containing `model.pt`
- **Model File**: `model.pt` - Pre-trained Chemprop model
- **Input/Output Files**: `logk_input` and `logk_output` for legacy interface
- **Descriptor Selection**: Uses RDKit descriptors 100-140 from the full 200 descriptor list
- **Feature Mask**: Boolean array determining which combined features to use for prediction

### Dependencies

**Core Runtime**:
- `rdkit` - Molecular processing and descriptor calculation
- `chemprop>=1.6.1` - Machine learning predictions
- `numpy`, `pandas` - Data handling
- `click` - CLI interface

**Development**:
- `pytest`, `black`, `isort`, `flake8`, `mypy` - Testing and code quality
- All configured in `pyproject.toml`