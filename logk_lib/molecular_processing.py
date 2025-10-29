"""
Molecular processing utilities for LOGKPREDICT.

This module provides functionality for:
- Calculating RDKit molecular descriptors
- Converting metal-ligand bonds to dative bonds
- Processing MOL block structures
- Generating cleaned SMILES representations
"""

import re
from typing import Tuple

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from .exceptions import MolecularProcessingError

# Constants for atomic numbers used in transition metal detection
HYDROGEN = 1
BORON_TO_FLUORINE_RANGE = (5, 9)
SILICON_TO_CHLORINE_RANGE = (14, 17)
BROMINE = 35

# Default donor atoms for dative bonds (Nitrogen, Oxygen)
DEFAULT_DONOR_ATOMS = (7, 8)

# RDKit descriptor indices used by LOGKPREDICT (100-140 from full descriptor list)
DESCRIPTOR_START_INDEX = 100
DESCRIPTOR_END_INDEX = 140


class MolecularProcessor:
    """Handles molecular processing operations for LOGKPREDICT."""

    # RDKit molecular descriptors (indices 100-140 of the full 200 descriptor list)
    SELECTED_DESCRIPTORS = [
        "PEOE_VSA1",
        "PEOE_VSA10",
        "PEOE_VSA11",
        "PEOE_VSA12",
        "PEOE_VSA13",
        "PEOE_VSA14",
        "PEOE_VSA2",
        "PEOE_VSA3",
        "PEOE_VSA4",
        "PEOE_VSA5",
        "PEOE_VSA6",
        "PEOE_VSA7",
        "PEOE_VSA8",
        "PEOE_VSA9",
        "RingCount",
        "SMR_VSA1",
        "SMR_VSA10",
        "SMR_VSA2",
        "SMR_VSA3",
        "SMR_VSA4",
        "SMR_VSA5",
        "SMR_VSA6",
        "SMR_VSA7",
        "SMR_VSA8",
        "SMR_VSA9",
        "SlogP_VSA1",
        "SlogP_VSA10",
        "SlogP_VSA11",
        "SlogP_VSA12",
        "SlogP_VSA2",
        "SlogP_VSA3",
        "SlogP_VSA4",
        "SlogP_VSA5",
        "SlogP_VSA6",
        "SlogP_VSA7",
        "SlogP_VSA8",
        "SlogP_VSA9",
        "TPSA",
        "VSA_EState1",
        # Need exactly 40 descriptors (indices 100-139)
        "VSA_EState10",
    ]

    def __init__(self) -> None:
        """Initialize the molecular processor."""
        self.descriptor_calculator = MolecularDescriptorCalculator(
            self.SELECTED_DESCRIPTORS
        )

    def calculate_descriptors(self, mol: Chem.Mol) -> str:
        """
        Calculate molecular descriptors using RDKit.

        Args:
            mol: RDKit molecule object

        Returns:
            Comma-separated string of descriptor values

        Raises:
            MolecularProcessingError: If descriptor calculation fails
        """
        try:
            descriptor_vals = list(self.descriptor_calculator.CalcDescriptors(mol))
            descriptor_strs = [str(val) for val in descriptor_vals]
            return ", ".join(descriptor_strs)
        except Exception as e:
            raise MolecularProcessingError(
                f"Failed to calculate molecular descriptors: {e}"
            )

    @staticmethod
    def is_transition_metal(atom: Chem.Atom) -> bool:
        """
        Determine if an atom is a transition metal.

        Excludes common non-transition elements: H, B-F, Si-Cl, and Br.

        Args:
            atom: RDKit atom object to analyze

        Returns:
            True if atom is likely a transition metal, False otherwise
        """
        atomic_num = atom.GetAtomicNum()

        is_hydrogen = atomic_num == HYDROGEN
        is_boron_to_fluorine = (
            BORON_TO_FLUORINE_RANGE[0] <= atomic_num <= BORON_TO_FLUORINE_RANGE[1]
        )
        is_silicon_to_chlorine = (
            SILICON_TO_CHLORINE_RANGE[0] <= atomic_num <= SILICON_TO_CHLORINE_RANGE[1]
        )
        is_bromine = atomic_num == BROMINE

        return not (
            is_hydrogen or is_boron_to_fluorine or is_silicon_to_chlorine or is_bromine
        )

    def set_dative_bonds(
        self, mol: Chem.Mol, donor_atoms: Tuple[int, ...] = DEFAULT_DONOR_ATOMS
    ) -> Chem.RWMol:
        """
        Convert metal-donor atom bonds to dative bonds.

        This is important for proper representation of metal complexes
        where nitrogen and oxygen atoms donate electron pairs to metals.

        Args:
            mol: Input RDKit molecule
            donor_atoms: Tuple of atomic numbers that can form dative bonds
                (default: N, O)

        Returns:
            RDKit RWMol object with dative bonds set

        Raises:
            MolecularProcessingError: If bond modification fails
        """
        try:
            rwmol = Chem.RWMol(mol)
            rwmol.UpdatePropertyCache(strict=False)

            metals = self._find_transition_metals(rwmol)
            self._convert_bonds_to_dative(rwmol, metals, donor_atoms)
            self._sanitize_molecule(rwmol)

            return rwmol
        except Exception as e:
            raise MolecularProcessingError(f"Failed to set dative bonds: {e}")

    def _find_transition_metals(self, mol: Chem.RWMol) -> list[Chem.Atom]:
        """Find all transition metal atoms in the molecule."""
        return [atom for atom in mol.GetAtoms() if self.is_transition_metal(atom)]

    def _convert_bonds_to_dative(
        self, mol: Chem.RWMol, metals: list[Chem.Atom], donor_atoms: Tuple[int, ...]
    ) -> None:
        """Convert bonds between metals and donor atoms to dative bonds."""
        for metal in metals:
            for neighbor in metal.GetNeighbors():
                if neighbor.GetAtomicNum() in donor_atoms:
                    mol.RemoveBond(neighbor.GetIdx(), metal.GetIdx())
                    mol.AddBond(neighbor.GetIdx(), metal.GetIdx(), Chem.BondType.DATIVE)

    def _sanitize_molecule(self, mol: Chem.RWMol) -> None:
        """Apply RDKit sanitization with specific flags for metal complexes."""
        sanitize_flags = (
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        )
        Chem.SanitizeMol(mol, sanitize_flags, catchErrors=True)

    @staticmethod
    def clean_smiles(smiles: str) -> str:
        """
        Remove redundant hydrogen annotations from SMILES string.

        This is specifically needed for dative bond representations
        where hydrogen atoms are sometimes redundantly included.

        Args:
            smiles: SMILES string to clean

        Returns:
            SMILES string with redundant hydrogens removed
        """
        # Pattern for removing hydrogens after metal atoms in dative bonds
        metal_hydrogen_pattern = r"(->\[[A-Z][a-z]?)(H\d?)"
        cleaned_smiles = re.sub(metal_hydrogen_pattern, r"\1", smiles)

        # Pattern for removing hydrogens before closing dative bond brackets
        hydrogen_bracket_pattern = r"(H\d?)(\+\d?\]<-)"
        cleaned_smiles = re.sub(hydrogen_bracket_pattern, r"\2", cleaned_smiles)

        return cleaned_smiles

    def process_mol_block(self, mol_block: str) -> Tuple[Chem.RWMol, str, str]:
        """
        Process a MOL block to create molecule with dative bonds and generate features.

        Args:
            mol_block: MOL format string

        Returns:
            Tuple of (processed_molecule, descriptors_string, smiles_string)

        Raises:
            MolecularProcessingError: If processing fails
        """
        try:
            # Parse MOL block
            mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
            if mol is None:
                raise MolecularProcessingError("Failed to parse MOL block")

            # Set dative bonds
            mol_with_dative = self.set_dative_bonds(mol)

            # Calculate descriptors
            descriptors = self.calculate_descriptors(mol_with_dative)

            # Generate SMILES with dative bonds
            smiles = Chem.MolToSmiles(mol_with_dative, allHsExplicit=False)
            clean_smiles = self.clean_smiles(smiles)

            return mol_with_dative, descriptors, clean_smiles

        except Exception as e:
            raise MolecularProcessingError(f"Failed to process MOL block: {e}")
