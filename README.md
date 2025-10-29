
# LOGKPREDICT

LOGKPREDICT is a computational chemistry tool that predicts stability constants (log K values) for metal-ligand complexes. It integrates [HostDesigner](https://sourceforge.net/projects/hostdesigner/) with [Chemprop](https://github.com/chemprop/chemprop) machine learning models to provide stability predictions for ranking designed molecules.

**Key Features:**
- Predicts log K stability constants using neural networks trained on experimental data
- Handles transition metal complexes with proper dative bond representation
- Calculates molecular descriptors using RDKit
- Clean, maintainable Python codebase following modern development practices
- Full compatibility with HostDesigner workflow

## Quick Start

The original interface remains unchanged for HostDesigner compatibility:
```bash
# Set environment variable
export LOGKPREDICT_DIR='/path/to/LOGKPREDICT/'

# Run prediction
./LOGKPREDICT
```

## New Python API

For advanced users and developers, a new Python API is available:
```python
from logk_lib import LogKPredictor

# Initialize predictor
predictor = LogKPredictor()

# Predict from file
result = predictor.predict_from_file('logk_input')
print(f'Predicted log K: {result}')

# Or predict directly from data
features = [0.0, 2.0, 0.699, 6.0, 3.01, 473.2, 1.3, 0.264]
mol_block = "..."
result = predictor.predict(features, mol_block)
```

## Installation

### Option 1: Modern Installation (Recommended)

1) **Create conda environment:**
```bash
# Create environment with Python and core dependencies
conda create -n logkpredict python=3.9 -y
conda activate logkpredict
conda install -c conda-forge rdkit numpy pandas -y

# Install Chemprop and other dependencies
pip install chemprop>=1.6.1 click typing-extensions

# For development (optional)
pip install pytest black isort flake8 mypy
```

2) **Install LOGKPREDICT:**
```bash
# Clone repository
git clone https://github.com/Critical-Materials-Institute/LOGKPREDICT.git
cd LOGKPREDICT

# Install in development mode (optional)
pip install -e .
```

### Option 2: Legacy Installation

For compatibility with older workflows:

1) Download chemprop-1.5.2+ from [chemprop repository](https://github.com/chemprop/chemprop)
2) Install chemprop following their instructions
3) Download LOGKPREDICT files


## Configuration

3) **Set up environment:**
```bash
# Make executable (if needed)
chmod 755 LOGKPREDICT

# Set environment variable
export LOGKPREDICT_DIR='/path/to/LOGKPREDICT/directory/'

# For permanent setup, add to ~/.bashrc or ~/.zshrc:
echo 'export LOGKPREDICT_DIR="/path/to/LOGKPREDICT/directory/"' >> ~/.bashrc
```

## Testing

Test the installation using the provided example files:
```bash
# Activate environment
conda activate logkpredict
export LOGKPREDICT_DIR='/path/to/LOGKPREDICT/'

# Test original interface
./LOGKPREDICT
# Should create logk_output with predicted value

# Test Python API
python -c "
from logk_lib import LogKPredictor
predictor = LogKPredictor()
result = predictor.predict_from_file('logk_input')
print(f'Predicted log K: {result}')
"
```

## Code Architecture

The codebase has been modernized while maintaining full backward compatibility:

```
LOGKPREDICT/
├── LOGKPREDICT              # Original executable (HostDesigner compatible)
├── model.pt                 # Pre-trained ML model
├── logk_input              # Example input file
├── logk_output             # Example output file
├── logk_lib/               # New Python library
│   ├── __init__.py        # Package interface
│   ├── exceptions.py      # Custom exception hierarchy
│   ├── molecular_processing.py  # RDKit molecular processing
│   └── predictor.py       # Main prediction logic
├── pyproject.toml         # Modern Python packaging
└── data/                  # Training datasets
```

### Key Improvements

- **Clean Code Principles**: Modular design with single responsibility
- **Type Safety**: Full type annotations and mypy compatibility
- **Error Handling**: Comprehensive exception hierarchy
- **Modern Python**: Uses pathlib, context managers, and latest practices
- **Developer Tools**: Pre-configured formatting, linting, and testing
- **Documentation**: Comprehensive docstrings and examples
