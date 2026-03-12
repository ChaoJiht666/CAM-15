# src/Local_Neighborhood_Construction.py
"""
CAM-15: Local Neighborhood Construction Module (v1.0.0)

Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Multi-scale neighborhood construction (Unigram/Bigram/Trigram)
- Enhanced neighborhood data structure with positional encoding
- Configurable window size and multi-scale processing
- Sinusoidal position encoding for sequential information
- Memory-efficient neighborhood representation

Key Improvements (v1.0.0):
1. Multi-scale neighborhood support (captures different granularity relationships)
2. Structured neighborhood data with relative positional information
3. Configurable multi-scale processing via system configuration
4. Sinusoidal position encoding for sequence awareness
"""

import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class NeighborhoodScale(Enum):
    """
    Enumeration for neighborhood scale (granularity levels).

    Defines the different scales of local neighborhoods to capture
    character relationships at varying granularities:
    - UNIGRAM: Single character (1-token context)
    - BIGRAM: Character pair (2-token context)
    - TRIGRAM: Character triplet (3-token context)
    """
    UNIGRAM = 1    # Single character context
    BIGRAM = 2     # Two character context (pair)
    TRIGRAM = 3    # Three character context (triplet)


@dataclass
class Neighborhood:
    """
    Enhanced neighborhood data structure with positional metadata.

    Represents a local neighborhood around a central character with
    comprehensive positional and scale information for relationship
    analysis in the CAM-15 model.

    Attributes:
        center_idx (int): Vocabulary index of the central character
        indices (List[int]): Vocabulary indices of all characters in neighborhood
        chars (List[str]): Actual character strings in the neighborhood
        position (int): Absolute position of center character in original sequence
        scale (NeighborhoodScale): Granularity scale (Unigram/Bigram/Trigram)
        relative_positions (List[int]): Relative positions to center character
                                        (negative=left, 0=center, positive=right)
    """
    center_idx: int
    indices: List[int]
    chars: List[str]
    position: int
    scale: NeighborhoodScale
    relative_positions: List[int]


class LocalNeighborhoodConstruction:
    """
    Core class for constructing multi-scale local neighborhoods from character sequences.

    This class builds structured neighborhood representations around each character
    in a sequence at multiple scales (Unigram/Bigram/Trigram) to capture
    character relationships at different granularities. It provides:
    - Configurable window size and multi-scale processing
    - Enhanced neighborhood metadata (positions, scales, relative offsets)
    - Sinusoidal position encoding for sequential information
    - Memory-efficient neighborhood construction

    Attributes:
        config (Dict): System configuration loaded from JSON file
        window_size (int): Base window size for neighborhood construction
        multi_scale (bool): Enable/disable multi-scale (Unigram/Bigram/Trigram) processing
        logger (logging.Logger): Module-specific logger instance
    """

    def __init__(self, config_path: str = "Config/System_Config.json"):
        """
        Initialize the LocalNeighborhoodConstruction instance.

        Args:
            config_path (str): Path to system configuration JSON file
                               Default: "Config/System_Config.json"

        Raises:
            FileNotFoundError: If config file does not exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        # Load system configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Configuration parameters with defaults
        self.window_size = self.config.get("neighborhood_size", 3)
        self.multi_scale = self.config.get("multi_scale", True)

        # Initialize module logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Setup module-specific logger with stream handler and formatted output.

        Returns:
            logging.Logger: Configured logger instance for CAM4.LNC
        """
        logger = logging.getLogger("CAM4.LNC")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if logger already initialized
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def construct_single(self, sequence: List[int], char_list: List[str],
                        pos: int, scale: NeighborhoodScale) -> Neighborhood:
        """
        Construct a single-scale neighborhood for a specific position in the sequence.

        Builds a structured neighborhood representation at the specified scale
        (Unigram/Bigram/Trigram) with padding for sequence boundaries and
        comprehensive positional metadata.

        Args:
            sequence (List[int]): Vocabulary indices of the input sequence
            char_list (List[str]): Corresponding character strings for the sequence
            pos (int): Position of center character in the sequence
            scale (NeighborhoodScale): Granularity scale for this neighborhood

        Returns:
            Neighborhood: Structured neighborhood object with complete metadata

        Notes:
            Uses <PAD> token (index 0) for sequence boundary padding
            Relative positions encode direction (negative=left, positive=right)
        """
        # Sequence length and padding configuration
        L = len(sequence)
        pad_idx = 0  # Vocabulary index for padding token

        # Trigram (3-token) neighborhood construction
        if scale == NeighborhoodScale.TRIGRAM:
            # Handle left boundary (padding for first position)
            if pos == 0:
                left_idx, left_char, left_rel = pad_idx, "<PAD>", -2
            else:
                left_idx, left_char, left_rel = sequence[pos-1], char_list[pos-1], -1

            # Center character (always present)
            center_idx, center_char, center_rel = sequence[pos], char_list[pos], 0

            # Handle right boundary (padding for last position)
            if pos == L - 1:
                right_idx, right_char, right_rel = pad_idx, "<PAD>", 2
            else:
                right_idx, right_char, right_rel = sequence[pos+1], char_list[pos+1], 1

            # Compile trigram neighborhood components
            indices = [left_idx, center_idx, right_idx]
            chars = [left_char, center_char, right_char]
            rel_pos = [left_rel, center_rel, right_rel]

        # Bigram (2-token) neighborhood construction
        elif scale == NeighborhoodScale.BIGRAM:
            # Center character (always present)
            center_idx = sequence[pos]
            center_char = char_list[pos]

            # Right neighbor (or padding)
            if pos < L - 1:
                right_idx = sequence[pos+1]
                right_char = char_list[pos+1]
                rel_pos = [0, 1]
            else:
                right_idx = pad_idx
                right_char = "<PAD>"
                rel_pos = [0, 2]

            # Compile bigram neighborhood components
            indices = [center_idx, right_idx]
            chars = [center_char, right_char]

        # Unigram (1-token) neighborhood construction
        else:
            # Single character (no context)
            indices = [sequence[pos]]
            chars = [char_list[pos]]
            rel_pos = [0]

        # Return fully structured neighborhood object
        return Neighborhood(
            center_idx=center_idx,
            indices=indices,
            chars=chars,
            position=pos,
            scale=scale,
            relative_positions=rel_pos
        )

    def construct(self, sequence: List[int], char_list: List[str]) -> List[Neighborhood]:
        """
        Construct multi-scale neighborhoods for the entire input sequence.

        Builds a comprehensive set of neighborhoods for each position in the sequence,
        including Trigram (primary) and optional Bigram (secondary) scales when
        multi-scale processing is enabled.

        Args:
            sequence (List[int]): Vocabulary indices of the input sequence
            char_list (List[str]): Corresponding character strings for the sequence

        Returns:
            List[Neighborhood]: Complete list of multi-scale neighborhoods
                                - Trigram neighborhoods for all positions
                                - Bigram neighborhoods (additional) when multi_scale=True

        Notes:
            Main entry point for neighborhood construction
            Returns Trigram neighborhoods for all positions + Bigram for internal positions
        """
        neighborhoods = []
        L = len(sequence)

        # Primary Trigram neighborhoods (all positions)
        for i in range(L):
            neigh = self.construct_single(sequence, char_list, i, NeighborhoodScale.TRIGRAM)
            neighborhoods.append(neigh)

            # Additional Bigram neighborhoods (multi-scale mode)
            if self.multi_scale and i < L - 1:
                bigram = self.construct_single(sequence, char_list, i, NeighborhoodScale.BIGRAM)
                neighborhoods.append(bigram)

        return neighborhoods

    def get_position_encoding(self, pos: int, max_len: int = 100) -> np.ndarray:
        """
        Generate sinusoidal position encoding for sequence positions.

        Creates 4-dimensional positional encoding vector using sine/cosine functions
        to capture sequential information for integration with CAM-15 features.

        Args:
            pos (int): Position index to encode
            max_len (int): Maximum sequence length for normalization
                           Default: 100

        Returns:
            np.ndarray: 4-dimensional positional encoding vector (float32)

        Notes:
            Follows transformer-style positional encoding (sine/cosine alternating)
            Fixed 4 dimensions to match CAM-4 baseline feature size
        """
        # Initialize 4-dimensional position encoding vector
        pe = np.zeros(4)

        # Generate sine/cosine positional encoding (alternating)
        for i in range(4):
            angle = pos / (10000 ** (2 * i / 4))
            pe[i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)

        return pe.astype(np.float32)


if __name__ == "__main__":
    # Example usage and verification
    lnc = LocalNeighborhoodConstruction()
    seq = [5, 8, 12, 3, 9]  # Example vocabulary indices
    chars = ["我", "爱", "自", "然", "语"]  # Corresponding characters

    # Construct multi-scale neighborhoods
    result = lnc.construct(seq, chars)

    # Log verification information
    print(f"CAM-15 Local Neighborhood Construction (v1.0.0)")
    print(f"Author: ChaoJiht666")
    print(f"Constructed {len(result)} neighborhoods (multi-scale enabled)")
    print(f"First neighborhood details: {result[0]}")
    print(f"Position encoding for position 2: {lnc.get_position_encoding(2)}")