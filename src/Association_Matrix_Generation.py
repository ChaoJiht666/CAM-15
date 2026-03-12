# src/Association_Matrix_Generation.py
"""
CAM-15: Association Matrix Generation Module (v1.0.0)

Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Generate association matrices for local neighborhood analysis
- Support multiple matrix storage formats with automatic detection
- Optimized with LRU cache for frequent matrix access
- Built-in normalization strategies and boundary checking

Supported Formats:
- .joblib.xz (LZMA compressed, minimal storage)
- .joblib (uncompressed joblib format)
- .npz (standard scipy sparse matrix format)
- _meta.json (metadata file with matrix path references)

Changelog v1.0.0:
- Initial release
- Added get_cooccurrence_value method for FeatureSequenceOutput support
- Fixed import handling for standalone execution
- Implemented automatic format detection and loading
"""

import json
import logging
import functools
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from scipy.sparse import load_npz, csr_matrix

# Optional dependency for compressed matrix loading
try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    joblib = None

# Local module import with fallback for standalone execution
try:
    from .Local_Neighborhood_Construction import Neighborhood, NeighborhoodScale
except ImportError:
    # Placeholder when relative import fails (direct script execution)
    Neighborhood = None
    NeighborhoodScale = None


class AssociationMatrixGeneration:
    """
    Core class for generating association matrices from precomputed co-occurrence matrices.

    This class handles loading co-occurrence matrices in various formats, provides
    cached access to matrix values, and generates association matrices with distance
    weighting for local neighborhood analysis.

    Attributes:
        config (Dict): System configuration loaded from JSON file
        logger (logging.Logger): Module-specific logger instance
        cooccur_matrix (csr_matrix): Loaded sparse co-occurrence matrix
        distance_decay (float): Decay factor for distance weighting (from config)
        vocab_size (int): Size of vocabulary (matrix dimension)
        _cache_size (int): LRU cache size for matrix access (from config)
        _enable_cache (bool): Flag to enable/disable caching (from config)
    """

    def __init__(self, config_path: str = "Config/System_Config.json"):
        """
        Initialize the AssociationMatrixGeneration instance.

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

        # Initialize logger
        self.logger = self._setup_logger()

        # Core matrix storage
        self.cooccur_matrix = None

        # Configuration parameters
        self.distance_decay = self.config.get("distance_decay", 0.5)
        self.vocab_size = 0

        # Cache configuration
        self._cache_size = self.config.get("matrix_cache_size", 1024)
        self._enable_cache = self.config.get("enable_matrix_cache", True)

    def _setup_logger(self) -> logging.Logger:
        """
        Setup module-specific logger with stream handler and formatted output.

        Returns:
            logging.Logger: Configured logger instance for CAM4.AMG
        """
        logger = logging.getLogger("CAM4.AMG")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_cooccurrence_matrix(self, matrix_path: str) -> None:
        """
        Load precomputed co-occurrence matrix with automatic format detection.

        Automatically identifies and loads matrices in different formats, including
        handling metadata JSON files that reference the actual matrix files.

        Args:
            matrix_path (str): Path to matrix file or metadata JSON file

        Raises:
            ImportError: If joblib is required but not installed
            FileNotFoundError: If matrix file cannot be found
            ValueError: If file format is unsupported or matrix data is invalid
        """
        matrix_path = Path(matrix_path)

        # Handle metadata JSON files
        if matrix_path.suffix == '.json' or matrix_path.name.endswith('_meta.json'):
            try:
                with open(matrix_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                # Try compressed path from metadata first
                compressed = meta.get('compressed_path')
                if compressed and Path(compressed).exists():
                    matrix_path = Path(compressed)
                else:
                    # Infer matrix file path from metadata filename
                    base_name = matrix_path.name.replace('_meta.json', '')
                    inferred_path = matrix_path.parent / f"{base_name}_matrix.joblib.xz"

                    if inferred_path.exists():
                        matrix_path = inferred_path
                    else:
                        # Fallback to standard NPZ format
                        npz_path = matrix_path.parent / f"{base_name}.npz"
                        if npz_path.exists():
                            matrix_path = npz_path
                        else:
                            raise FileNotFoundError(
                                f"Metadata file exists but no corresponding matrix file found: {matrix_path}"
                            )
            except Exception as e:
                self.logger.error(f"Failed to parse metadata file: {str(e)}")
                raise

        # Determine loading method based on file extension
        suffix = matrix_path.suffix.lower()
        full_name = matrix_path.name.lower()

        # Handle LZMA compressed joblib files (.joblib.xz)
        if full_name.endswith('.joblib.xz') or full_name.endswith('.xz'):
            if not HAS_JOBLIB:
                raise ImportError("joblib library is required to load XZ compressed matrices")

            self.logger.info(f"Loading LZMA compressed matrix: {matrix_path}")
            try:
                # Load and decompress matrix data
                matrix_data = joblib.load(matrix_path)

                # Reconstruct CSR matrix from components if needed
                if isinstance(matrix_data, tuple) and len(matrix_data) == 4:
                    data, indices, indptr, shape = matrix_data
                    self.cooccur_matrix = csr_matrix(
                        (data.astype(np.float32), indices, indptr),
                        shape=shape,
                        dtype=np.float32
                    )
                elif isinstance(matrix_data, csr_matrix):
                    # Direct CSR matrix object
                    self.cooccur_matrix = matrix_data.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported joblib matrix format: {type(matrix_data)}")

            except Exception as e:
                self.logger.error(f"Failed to load joblib.xz matrix: {str(e)}")
                raise

        # Handle uncompressed joblib/pickle files
        elif suffix in ('.joblib', '.pkl', '.pickle'):
            if not HAS_JOBLIB:
                raise ImportError("joblib library is required to load joblib/pickle matrices")

            self.logger.info(f"Loading joblib matrix: {matrix_path}")
            matrix_data = joblib.load(matrix_path)

            # Reconstruct CSR matrix from components if needed
            if isinstance(matrix_data, tuple) and len(matrix_data) == 4:
                data, indices, indptr, shape = matrix_data
                self.cooccur_matrix = csr_matrix(
                    (data.astype(np.float32), indices, indptr),
                    shape=shape,
                    dtype=np.float32
                )
            elif isinstance(matrix_data, csr_matrix):
                self.cooccur_matrix = matrix_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported joblib matrix format: {type(matrix_data)}")

        # Handle standard SciPy NPZ format
        elif suffix == '.npz':
            self.logger.info(f"Loading standard SciPy matrix: {matrix_path}")
            try:
                self.cooccur_matrix = load_npz(str(matrix_path)).astype(np.float32)
            except ValueError as e:
                if "pickled" in str(e).lower():
                    # Handle NPZ files with pickle data (allow_pickle=True required)
                    self.logger.warning("Detected NPZ with pickle data - loading with allow_pickle=True")
                    loaded = np.load(str(matrix_path), allow_pickle=True)
                    # Manually reconstruct CSR matrix
                    self.cooccur_matrix = csr_matrix(
                        (loaded['data'], loaded['indices'], loaded['indptr']),
                        shape=loaded['shape'],
                        dtype=np.float32
                    )
                else:
                    raise

        # Unsupported format
        else:
            raise ValueError(f"Unsupported matrix file format: {suffix}")

        # Update vocabulary size and log matrix stats
        self.vocab_size = self.cooccur_matrix.shape[0]
        self.logger.info(
            f"Co-occurrence matrix loaded successfully: "
            f"Shape={self.cooccur_matrix.shape}, Non-zero elements={self.cooccur_matrix.nnz}"
        )

    @functools.lru_cache(maxsize=1024)
    def _get_cached_cooccurrence(self, idx_i: int, idx_j: int) -> float:
        """
        LRU cache wrapper for frequent matrix access operations.

        Private method - should not be called directly by external code.

        Args:
            idx_i (int): Row index in co-occurrence matrix
            idx_j (int): Column index in co-occurrence matrix

        Returns:
            float: Co-occurrence value at (idx_i, idx_j) or 0.0 if invalid
        """
        if self.cooccur_matrix is None:
            return 0.0
        if idx_i < 0 or idx_i >= self.vocab_size or idx_j < 0 or idx_j >= self.vocab_size:
            return 0.0
        return float(self.cooccur_matrix[idx_i, idx_j])

    def get_cooccurrence_value(self, idx_i: int, idx_j: int) -> float:
        """
        Get single co-occurrence value with optional caching and boundary checking.

        Main public interface for accessing individual matrix values with safety checks.

        Args:
            idx_i (int): Row index in co-occurrence matrix
            idx_j (int): Column index in co-occurrence matrix

        Returns:
            float: Co-occurrence value at specified indices (0.0 if out of bounds)
        """
        if not self._enable_cache:
            # Non-cached mode (direct access)
            if self.cooccur_matrix is None:
                return 0.0
            if idx_i < 0 or idx_i >= self.vocab_size or idx_j < 0 or idx_j >= self.vocab_size:
                return 0.0
            return float(self.cooccur_matrix[idx_i, idx_j])

        # Use LRU cache for optimized access
        return self._get_cached_cooccurrence(int(idx_i), int(idx_j))

    def generate(self, neighborhood) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate association matrix and direction mask for a single neighborhood.

        Creates a dense association matrix with distance decay weighting and a
        direction mask (upper triangle = forward, lower triangle = backward).

        Args:
            neighborhood: Neighborhood object/dict/list/tuple containing indices
                          - Object with 'indices' attribute
                          - Dictionary with 'indices' key
                          - List/tuple of indices

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Association matrix (size x size, float32)
                - Direction mask (size x size, int8)

        Raises:
            ValueError: If co-occurrence matrix not loaded or invalid neighborhood type
        """
        if self.cooccur_matrix is None:
            raise ValueError("Co-occurrence matrix not loaded - call load_cooccurrence_matrix() first")

        # Extract indices from different neighborhood representations
        if hasattr(neighborhood, 'indices'):
            indices = neighborhood.indices
        elif isinstance(neighborhood, dict):
            indices = neighborhood.get('indices', [])
        elif isinstance(neighborhood, (list, tuple)):
            indices = neighborhood
        else:
            raise ValueError(f"Unsupported neighborhood type: {type(neighborhood)}")

        size = len(indices)
        if size == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int8)

        # Initialize association matrix
        A = np.zeros((size, size), dtype=np.float32)

        # Populate matrix with distance-weighted co-occurrence values
        for m in range(size):
            for n in range(size):
                base_val = self.get_cooccurrence_value(indices[m], indices[n])
                dist = abs(m - n)
                weight = np.exp(-self.distance_decay * dist)
                A[m, n] = base_val * weight

        # Generate direction mask (upper=1, lower=-1, diagonal=0)
        mask = np.triu(np.ones((size, size), dtype=np.int8), k=1) - \
               np.tril(np.ones((size, size), dtype=np.int8), k=-1)

        return A, mask

    def generate_batch(self, neighborhoods: List) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch generate association matrices and masks for multiple neighborhoods.

        Args:
            neighborhoods (List): List of neighborhood objects/structures

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (matrix, mask) tuples
        """
        results = []
        for neigh in neighborhoods:
            A, mask = self.generate(neigh)
            results.append((A, mask))
        return results

    def normalize(self, A: np.ndarray, method: str = "softmax") -> np.ndarray:
        """
        Normalize association matrix using specified method.

        Args:
            A (np.ndarray): Input association matrix
            method (str): Normalization method ('softmax', 'standard', 'minmax')
                          Default: 'softmax'

        Returns:
            np.ndarray: Normalized matrix (same shape as input)

        Notes:
            - softmax: Exponential normalization with max subtraction for stability
            - standard: Z-score normalization (mean=0, std=1)
            - minmax: Rescales to [0, 1] range (default for unknown methods)
        """
        eps = np.float32(1e-10)  # Prevent division by zero

        if method == "softmax":
            # Softmax normalization (numerically stable)
            exp_A = np.exp(A - np.max(A))
            return exp_A / (np.sum(exp_A) + eps)
        elif method == "standard":
            # Standard (Z-score) normalization
            mean = np.mean(A)
            std = np.std(A)
            return (A - mean) / (std + eps)
        else:
            # Min-max normalization (default fallback)
            min_val = np.min(A)
            max_val = np.max(A)
            return (A - min_val) / (max_val - min_val + eps)


if __name__ == "__main__":
    # Simple test/initialization when run as standalone script
    amg = AssociationMatrixGeneration()
    print("CAM-15 Association Matrix Generation Module (v1.0.0) ready")
    print("Author: ChaoJiht666")