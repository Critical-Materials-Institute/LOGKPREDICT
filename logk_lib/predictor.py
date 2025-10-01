"""
Stability constant prediction using machine learning.

This module provides the main prediction functionality for LOGKPREDICT,
integrating molecular processing with Chemprop neural network models.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .exceptions import (
    ChempropError,
    EnvironmentError,
    InvalidInputError,
    ModelNotFoundError,
)
from .molecular_processing import MolecularProcessor

# Constants
ENVIRONMENT_VARIABLE_NAME = "LOGKPREDICT_DIR"
MODEL_FILENAME = "model.pt"
INPUT_FILENAME = "logk_input"
OUTPUT_FILENAME = "logk_output"

# Feature names expected in input
EXPECTED_FEATURES = [
    "logK(I=0.0)",
    "logK_in",
    "I_in",
    "Z_lig",
    "Z_met",
    "nrot",
    "met_r",
    "met_CN",
    "E_strain",
    "G_solv",
    "rdhE",
    "rdhC",
]

# Feature mask for model compatibility (directly from working LOGKPREDICT)
_MASK_STR = "False  True False False  True  True  True  True  True  True  True False False  True  True  True  True False  True  True  True  True False  True True True False False False False False False False False True False False False False False False True False False False False False False False False"
FEATURE_MASK = np.array([eval(x) for x in _MASK_STR.split()])


class LogKPredictor:
    """Predicts stability constants (log K) for metal-ligand complexes."""

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize predictor with model directory.

        Args:
            model_dir: Directory containing model.pt file.
                      If None, uses LOGKPREDICT_DIR environment variable.

        Raises:
            EnvironmentError: If model directory cannot be determined
            ModelNotFoundError: If model file is not found
        """
        self.model_dir = self._determine_model_directory(model_dir)
        self.model_path = self._validate_model_exists()
        self.molecular_processor = MolecularProcessor()

    def _determine_model_directory(self, model_dir: Optional[str]) -> Path:
        """Determine the model directory from parameter or environment."""
        if model_dir:
            return Path(model_dir)

        env_dir = os.environ.get(ENVIRONMENT_VARIABLE_NAME)
        if not env_dir:
            raise EnvironmentError(
                f"Model directory not provided and {ENVIRONMENT_VARIABLE_NAME} "
                "environment variable not set"
            )

        return Path(env_dir)

    def _validate_model_exists(self) -> Path:
        """Validate that the model file exists."""
        model_path = self.model_dir / MODEL_FILENAME
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        return model_path

    def predict_from_file(self, input_file: str = INPUT_FILENAME) -> float:
        """
        Predict log K from input file.

        Args:
            input_file: Path to input file containing molecular data

        Returns:
            Predicted log K value

        Raises:
            InvalidInputError: If input file is malformed
            ChempropError: If prediction fails
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise InvalidInputError(f"Input file not found: {input_file}")

        with open(input_path) as file:
            lines = file.readlines()

        if len(lines) < 3:
            raise InvalidInputError(
                "Input file must contain at least header, features, and MOL block"
            )

        features = self._parse_features(lines[1])
        mol_block = self._extract_mol_block(lines[2:])

        return self.predict(features, mol_block)

    def predict(self, features: List[float], mol_block: str) -> float:
        """
        Predict log K from features and molecular structure.

        Args:
            features: List of numerical features
            mol_block: MOL format molecular structure

        Returns:
            Predicted log K value

        Raises:
            ChempropError: If prediction fails
        """
        # Process molecular structure
        _, descriptors_str, smiles = self.molecular_processor.process_mol_block(
            mol_block
        )

        # Combine features with molecular descriptors
        combined_features = self._combine_features(features, descriptors_str)

        # Create temporary files for Chemprop
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            input_csv = temp_path / "input.csv"
            features_csv = temp_path / "features.csv"
            predictions_csv = temp_path / "predictions.csv"

            self._create_input_files(smiles, combined_features, input_csv, features_csv)
            self._run_chemprop_prediction(input_csv, features_csv, predictions_csv)

            return self._extract_prediction(predictions_csv)

    def _parse_features(self, feature_line: str) -> List[float]:
        """Parse numerical features from input line."""
        try:
            # Remove newline and split
            feature_values = feature_line.strip().split()

            # Use same logic as newer LOGKPREDICT: features0[2:]
            # This gives us all features from I_in onward
            selected_features = feature_values[2:]
            return [float(val) for val in selected_features]
        except (ValueError, IndexError) as e:
            raise InvalidInputError(f"Failed to parse features: {e}")

    def _extract_mol_block(self, lines: List[str]) -> str:
        """Extract MOL block from input lines."""
        # Find the end marker
        mol_lines = []
        for line in lines:
            if line.strip() == "$$$$":
                break
            mol_lines.append(line)

        if not mol_lines:
            raise InvalidInputError("No MOL block found in input")

        return "".join(mol_lines)

    def _combine_features(self, features: List[float], descriptors_str: str) -> str:
        """Combine input features with molecular descriptors."""
        # Round features to 4 decimal places (as in original)
        rounded_features = [round(f, 4) for f in features]
        feature_str = ", ".join(map(str, rounded_features))

        # Combine with molecular descriptors
        combined_str = f"{feature_str}, {descriptors_str}"

        # Apply feature mask for model compatibility
        all_features = np.array([float(x) for x in combined_str.split(", ")])
        masked_features = all_features[FEATURE_MASK]

        return ", ".join(map(str, masked_features))

    def _create_input_files(
        self, smiles: str, features: str, input_csv: Path, features_csv: Path
    ) -> None:
        """Create CSV files needed by Chemprop."""
        # Create SMILES input file
        with open(input_csv, "w") as file:
            file.write("smiles\n")
            file.write(f"{smiles}\n")

        # Create features file with exact header from working LOGKPREDICT
        feature_header = (
            "I_in, Z_lig, Z_met, nrot, met_r, met_CN, E_strain, G_solv, rdhE, rdhC"
        )

        with open(features_csv, "w") as file:
            file.write(f"{feature_header}\n")
            file.write(f"{features}\n")

    def _run_chemprop_prediction(
        self, input_csv: Path, features_csv: Path, predictions_csv: Path
    ) -> None:
        """Execute Chemprop prediction command."""
        command = [
            "chemprop_predict",
            "--num_workers",
            "0",
            "--test_path",
            str(input_csv),
            "--features_path",
            str(features_csv),
            "--checkpoint_path",
            str(self.model_path),
            "--preds_path",
            str(predictions_csv),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ChempropError(f"Chemprop prediction failed: {e.stderr}")
        except FileNotFoundError:
            raise ChempropError(
                "chemprop_predict command not found. "
                "Please ensure Chemprop is installed and in PATH."
            )

    def _extract_prediction(self, predictions_csv: Path) -> float:
        """Extract prediction value from Chemprop output."""
        try:
            df = pd.read_csv(predictions_csv)
            if df.empty or len(df.columns) < 2:
                raise ChempropError("Invalid prediction output format")

            # Extract the first prediction value
            prediction = float(df.iloc[0, 1])
            return prediction

        except (pd.errors.EmptyDataError, ValueError, IndexError) as e:
            raise ChempropError(f"Failed to parse prediction output: {e}")
