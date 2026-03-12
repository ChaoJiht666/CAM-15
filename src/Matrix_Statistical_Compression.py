# src/Matrix_Statistical_Compression.py
"""
CAM-15: Matrix Statistical Compression Module (v1.0.0)

Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- 15-dimensional feature compression (solves CAM-4 information bottleneck)
- Unified 3×3 matrix processing (supports 1×1/2×2/3×3 input matrices)
- Structured feature representation with interpretability
- Batch processing for efficient matrix compression
- Feature explainability analysis for model interpretability

Key Improvements (v1.0.0):
1. Expanded from 4D to 15D feature space (eliminates information bottleneck)
2. Unified 3×3 matrix padding (consistent processing for all input sizes)
3. Structured feature dataclass with type safety
4. Batch processing pipeline for high-throughput compression
5. Feature interpretability analysis for model debugging
"""

import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CAM15Feature:
    """
    Structured CAM-15 feature vector representation (15-dimensional).
    
    Enhanced feature structure that expands from CAM-4 (4D) to CAM-15 (15D)
    to eliminate information bottlenecks while maintaining interpretability.
    Combines structural matrix elements, statistical measures, and positional
    encoding for comprehensive relationship representation.
    
    Core Feature Groups:
    1. Structural elements (6 features): Direct matrix element values (central relationships)
    2. Statistical measures (4 features): Trace, max, mean, std of matrix values
    3. Contextual features (2 features): Asymmetry and concentration metrics
    4. Positional encoding (3 features): Spatial/sequential information
    
    Attributes:
        a00 (float): Center self-correlation (A[1,1] in 3×3 matrix)
        a01 (float): Center → right forward link (A[1,2])
        a02 (float): Center → left backward link (A[1,0])
        a11 (float): Right element self-correlation (A[2,2])
        a12 (float): Right → left cross link (A[2,0])
        a22 (float): Left element self-correlation (A[0,0])
        
        trace (float): Matrix trace (sum of diagonal elements)
        max_val (float): Maximum value in matrix
        mean (float): Mean value of matrix elements
        std (float): Standard deviation of matrix elements
        
        asymmetry (float): Forward/backward asymmetry (-1 to +1)
        concentration (float): Value concentration (max/mean ratio)
        
        pos_enc (Optional[np.ndarray]): 3-dimensional positional encoding
                                        Default: None (zero-vector)
    """
    a00: float
    a01: float
    a02: float
    a11: float
    a12: float
    a22: float
    trace: float
    max_val: float
    mean: float
    std: float
    asymmetry: float
    concentration: float
    pos_enc: Optional[np.ndarray] = None

    def to_vector(self) -> np.ndarray:
        """
        Convert structured CAM15Feature to 15-dimensional numpy vector (float32).
        
        Combines structural elements (12 features) with positional encoding (3 features)
        to create the complete 15-dimensional CAM-15 feature vector.
        
        Returns:
            np.ndarray: 15-dimensional feature vector (float32)
                        [a00, a01, a02, a11, a12, a22, trace, max_val, mean, std,
                         asymmetry, concentration, pos_enc[0], pos_enc[1], pos_enc[2]]
        
        Notes:
            Uses zero-vector padding if positional encoding is not provided
            All values cast to float32 for memory efficiency
        """
        # Base 12-dimensional feature vector (structural + statistical)
        base = np.array([
            self.a00, self.a01, self.a02, self.a11, self.a12, self.a22,
            self.trace, self.max_val, self.mean, self.std,
            self.asymmetry, self.concentration
        ], dtype=np.float32)

        # Add 3-dimensional positional encoding (or zero padding)
        if self.pos_enc is not None:
            pos_part = self.pos_enc[:3].astype(np.float32)
        else:
            pos_part = np.zeros(3, dtype=np.float32)
            
        # Combine to complete 15-dimensional vector
        return np.concatenate([base, pos_part])


class MatrixStatisticalCompression:
    """
    Core implementation of CAM-15 compression operator for association matrices.
    
    This class provides the central compression logic for the CAM-15 model,
    converting association matrices of varying sizes (1×1, 2×2, 3×3) into
    a standardized 15-dimensional feature vector. Key capabilities:
    - Unified 3×3 matrix processing with intelligent padding
    - 15D feature extraction (solves CAM-4 information bottleneck)
    - Batch processing for high-throughput compression
    - Feature interpretability analysis
    - Memory-efficient float32 processing
    
    Attributes:
        config (Dict): System configuration loaded from JSON file
        eps (float): Small epsilon value to prevent division by zero
        logger (logging.Logger): Module-specific logger instance
        feature_dim (int): Fixed feature dimension (15 for CAM-15)
    """

    def __init__(self, config_path: str = "Config/System_Config.json"):
        """
        Initialize the MatrixStatisticalCompression instance.
        
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

        # Configuration parameters with safety defaults
        self.eps = self.config.get("epsilon", 1e-10)  # Numerical stability
        self.logger = self._setup_logger()
        self.feature_dim = 15  # Fixed for CAM-15

    def _setup_logger(self) -> logging.Logger:
        """
        Setup module-specific logger with stream handler and formatted output.
        
        Returns:
            logging.Logger: Configured logger instance for CAM4.MSC
        """
        logger = logging.getLogger("CAM4.MSC")
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

    def _pad_matrix_to_3x3(self, A: np.ndarray) -> np.ndarray:
        """
        Pad arbitrary square matrix (1×1/2×2) to standard 3×3 format.
        
        Intelligent padding that centers smaller matrices in the 3×3 grid
        to maintain consistent feature extraction regardless of input size.
        
        Args:
            A (np.ndarray): Input square matrix (1×1, 2×2, or 3×3)
        
        Returns:
            np.ndarray: 3×3 padded matrix (float32) with original values centered
            
        Padding Layout Examples:
            1×1 [x] → [[0,0,0],
                       [0,x,0],
                       [0,0,0]]
                       
            2×2 [[a,b], → [[0,0,0],
                 [c,d]]    [0,a,b],
                          [0,c,d]]
                          
            3×3 → returned unchanged
        """
        # Return 3×3 matrices as-is
        if A.shape == (3, 3):
            return A.astype(np.float32)

        # Initialize zero-padded 3×3 matrix (float32 for memory efficiency)
        padded = np.zeros((3, 3), dtype=np.float32)
        rows, cols = A.shape

        # Calculate center-aligned padding offsets
        row_start = (3 - rows) // 2
        col_start = (3 - cols) // 2
        
        # Place original matrix in center of padded matrix
        padded[row_start:row_start+rows, col_start:col_start+cols] = A.astype(np.float32)

        return padded

    def compress(self, A: np.ndarray, pos_enc: Optional[np.ndarray] = None) -> CAM15Feature:
        """
        Compress association matrix to structured CAM-15 feature representation.
        
        Core compression method that:
        1. Normalizes input to 3×3 matrix via padding
        2. Extracts structural matrix elements (central relationships)
        3. Calculates statistical measures (trace, max, mean, std)
        4. Computes contextual features (asymmetry, concentration)
        5. Incorporates positional encoding (if provided)
        
        Args:
            A (np.ndarray): Input association matrix (1×1, 2×2, or 3×3)
            pos_enc (Optional[np.ndarray]): Positional encoding vector
                                            Default: None (zero-vector)
        
        Returns:
            CAM15Feature: Structured 15-dimensional feature representation
            
        Key Matrix Mapping (3×3 layout):
            [0,0] = Top-left (left element self-correlation)
            [1,1] = Center (central element self-correlation)
            [2,2] = Bottom-right (right element self-correlation)
            [1,0] = Center-left (backward link)
            [1,2] = Center-right (forward link)
            [2,0] = Bottom-left (cross link)
        """
        # Normalize all inputs to 3×3 matrix format
        if A.shape != (3, 3):
            A = self._pad_matrix_to_3x3(A)

        # 1. Extract structural matrix elements (central relationship features)
        # Core relationship features (6 dimensions)
        a00 = A[1, 1]  # Center self-correlation (primary feature)
        a01 = A[1, 2]  # Center → right (forward directional link)
        a02 = A[1, 0]  # Center → left (backward directional link)
        a11 = A[2, 2]  # Right element self-correlation
        a12 = A[2, 0]  # Right → left (cross directional link)
        a22 = A[0, 0]  # Left element self-correlation

        # 2. Calculate statistical matrix measures (4 dimensions)
        trace = np.trace(A)          # Sum of diagonal elements (structural strength)
        max_val = np.max(A)          # Maximum relationship strength
        mean = np.mean(A)            # Average relationship strength
        std = np.std(A)              # Variability of relationship strengths

        # 3. Compute contextual structural features (2 dimensions)
        forward = a01                # Forward flow strength
        backward = a02               # Backward flow strength
        
        # Asymmetry: normalized forward/backward imbalance (-1 to +1)
        asymmetry = (forward - backward) / (abs(forward) + abs(backward) + self.eps)
        
        # Concentration: measure of value concentration (higher = more focused)
        concentration = max_val / (mean + self.eps)

        # 4. Prepare positional encoding (3 dimensions)
        pos_enc_3d = pos_enc[:3] if pos_enc is not None else np.zeros(3, dtype=np.float32)

        # Return structured CAM-15 feature object
        return CAM15Feature(
            a00=float(a00), a01=float(a01), a02=float(a02),
            a11=float(a11), a12=float(a12), a22=float(a22),
            trace=float(trace), max_val=float(max_val),
            mean=float(mean), std=float(std),
            asymmetry=float(asymmetry),
            concentration=float(concentration),
            pos_enc=pos_enc_3d
        )

    def compress_batch(self, matrices: List[np.ndarray],
                      pos_encs: Optional[List[np.ndarray]] = None) -> List[CAM15Feature]:
        """
        Batch compression of multiple association matrices.
        
        Efficient batch processing for high-throughput feature extraction
        with optional positional encoding for each matrix.
        
        Args:
            matrices (List[np.ndarray]): List of input matrices (1×1/2×2/3×3)
            pos_encs (Optional[List[np.ndarray]]): List of positional encodings
                                                   Default: None (zero-vectors)
        
        Returns:
            List[CAM15Feature]: List of structured CAM-15 features (one per matrix)
        
        Notes:
            Automatically handles missing positional encodings with zero padding
            Maintains input/output order correspondence
        """
        # Initialize positional encodings with None if not provided
        if pos_encs is None:
            pos_encs = [None] * len(matrices)

        # Compress each matrix in batch
        features = []
        for A, pe in zip(matrices, pos_encs):
            feat = self.compress(A, pe)
            features.append(feat)
            
        return features

    def compress_to_numpy(self, matrices: List[np.ndarray],
                         pos_encs: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Batch compression to numpy array (high-performance interface).
        
        Converts list of matrices directly to 2D numpy array of features
        (N × 15) for direct integration with machine learning pipelines.
        
        Args:
            matrices (List[np.ndarray]): List of input matrices (1×1/2×2/3×3)
            pos_encs (Optional[List[np.ndarray]]): List of positional encodings
                                                   Default: None (zero-vectors)
        
        Returns:
            np.ndarray: 2D feature array with shape (N, 15) (float32)
                        N = number of matrices, 15 = CAM-15 feature dimensions
        
        Notes:
            Optimized for ML pipeline integration
            All values cast to float32 for memory efficiency
        """
        # Compress to structured features
        features = self.compress_batch(matrices, pos_encs)
        
        # Convert to 2D numpy array (float32)
        vectors = np.stack([f.to_vector() for f in features])
        
        return vectors

    def explain_feature(self, feat: CAM15Feature) -> Dict[str, str]:
        """
        Generate human-readable feature interpretation for debugging/analysis.
        
        Converts numerical CAM-15 features into interpretable text descriptions
        grouped by feature category (structure, statistics, context).
        
        Args:
            feat (CAM15Feature): Structured CAM-15 feature to interpret
        
        Returns:
            Dict[str, str]: Interpretability dictionary with:
                            - structure: Core relationship strengths
                            - stats: Statistical matrix properties
                            - context: Contextual flow characteristics
        
        Notes:
            Values rounded to 2-3 decimal places for readability
            Asymmetry prefixed with +/- for intuitive interpretation
        """
        return {
            "structure": f"center:{feat.a00:.3f}, fwd:{feat.a01:.3f}, bwd:{feat.a02:.3f}",
            "stats": f"trace:{feat.trace:.3f}, max:{feat.max_val:.3f}, mean:{feat.mean:.3f}, std:{feat.std:.3f}",
            "context": f"asym:{feat.asymmetry:+.3f}, conc:{feat.concentration:.2f}"
        }


if __name__ == "__main__":
    # Example usage and verification
    msc = MatrixStatisticalCompression()
    
    # Test matrix (2×2 bigram example)
    test_matrix = np.array([[0.8, 0.6], [0.4, 0.9]], dtype=np.float32)
    
    # Single matrix compression
    feature = msc.compress(test_matrix)
    vector = feature.to_vector()
    
    # Batch compression example
    batch_matrices = [test_matrix, np.array([[0.5]]), np.eye(3)]
    batch_features = msc.compress_to_numpy(batch_matrices)
    
    # Log verification information
    print(f"CAM-15 Matrix Statistical Compression (v1.0.0)")
    print(f"Author: ChaoJiht666")
    print(f"Single feature vector shape: {vector.shape} (expected: (15,))")
    print(f"Batch feature array shape: {batch_features.shape} (expected: (3, 15))")
    print(f"Feature interpretation:\n{msc.explain_feature(feature)}")
    print(f"Successfully expanded from CAM-4 (4D) to CAM-15 (15D) - no information bottleneck")