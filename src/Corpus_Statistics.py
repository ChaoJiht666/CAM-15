# src/Corpus_Statistics.py
"""
CAM-15: Corpus Statistics Module (v1.0.0)

Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Character frequency analysis for Chinese text corpora
- Automatic Laplace smoothing coefficient calculation
- Batch processing of CSV files with text data
- Serialization/deserialization of statistical results
- Comprehensive logging and progress tracking

Use Cases:
- Preprocessing step for cooccurrence matrix construction
- Parameter tuning for Laplace smoothing
- Corpus quality assessment and analysis
"""

import json
import logging
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class CorpusStatistics:
    """
    Core class for statistical analysis of Chinese text corpora.

    This class provides comprehensive character-level statistics for text corpora
    stored in CSV files, including frequency counting, total character counting,
    and automatic calculation of optimal Laplace smoothing coefficients. Results
    can be saved/loaded for reuse in downstream processing pipelines.

    Attributes:
        config (Dict): System configuration loaded from JSON file
        logger (logging.Logger): Module-specific logger instance
        char_frequency (Counter): Character frequency counter (char -> count)
        total_chars (int): Total number of characters processed
        laplace_alpha (float): Calculated optimal Laplace smoothing coefficient
    """

    def __init__(self, config_path: str = "Config/System_Config.json"):
        """
        Initialize the CorpusStatistics instance.

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

        # Initialize logger and core statistics
        self.logger = self._setup_logger()
        self.char_frequency = Counter()
        self.total_chars = 0
        self.laplace_alpha = 1.0  # Default smoothing coefficient

    def _setup_logger(self) -> logging.Logger:
        """
        Setup module-specific logger with stream handler and formatted output.

        Returns:
            logging.Logger: Configured logger instance for CAM4.CS
        """
        logger = logging.getLogger("CAM4.CS")
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

    def process_file(self, file_path: str) -> None:
        """
        Process a single CSV file to extract character-level statistics.

        Reads text data from 'text' column, counts character frequencies, and
        accumulates total character count. Handles both UTF-8 and GBK encodings.

        Args:
            file_path (str): Path to CSV file containing text data

        Raises:
            FileNotFoundError: If CSV file does not exist
            KeyError: If CSV file does not contain 'text' column
        """
        try:
            # Read CSV with UTF-8 encoding (primary)
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to GBK encoding for Chinese text files
            df = pd.read_csv(file_path, encoding='gbk')

        # Extract and process text data
        texts = df['text'].astype(str).tolist()

        # Process each text with progress tracking
        for text in tqdm(texts, desc=f"Processing {Path(file_path).name}"):
            if not text.strip():
                continue

            # Split text into individual characters and update statistics
            chars = list(text.strip())
            self.char_frequency.update(chars)
            self.total_chars += len(chars)

    def process_directory(self, data_dir: str) -> Dict:
        """
        Process all CSV files in a directory to compute corpus-wide statistics.

        Scans directory for CSV files, processes each file sequentially, and
        calculates optimal Laplace smoothing coefficient based on corpus statistics.

        Args:
            data_dir (str): Path to directory containing CSV files with text data

        Returns:
            Dict: Comprehensive corpus statistics including:
                - char_frequency: Character frequency counter
                - total_chars: Total characters processed
                - unique_chars: Number of unique characters
                - laplace_alpha: Calculated smoothing coefficient
                - top_10_chars: Top 10 most frequent characters

        Notes:
            Laplace smoothing coefficient calculated as sqrt(total_chars / unique_chars)
            Provides balanced smoothing for cooccurrence matrix construction
        """
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))

        # Handle empty directory case
        if not csv_files:
            self.logger.warning(f"No CSV files found in directory: {data_dir}")
            return {}

        # Log processing parameters
        self.logger.info(f"Starting corpus statistics analysis - found {len(csv_files)} CSV files")

        # Process each CSV file in the directory
        for csv_file in csv_files:
            self.logger.info(f"Processing file: {csv_file.name}")
            self.process_file(str(csv_file))

        # Calculate optimal Laplace smoothing coefficient
        unique_chars = len(self.char_frequency)
        if unique_chars > 0:
            # Optimal alpha: square root of (total chars / unique chars)
            self.laplace_alpha = np.sqrt(self.total_chars / unique_chars)
        else:
            self.laplace_alpha = 1.0  # Fallback for empty corpus

        # Log summary statistics
        self.logger.info(f"Corpus statistics analysis complete: {unique_chars} unique characters")
        self.logger.info(f"Recommended Laplace smoothing coefficient: {self.laplace_alpha:.2f}")

        # Compile comprehensive statistics dictionary
        stats = {
            "char_frequency": self.char_frequency,
            "total_chars": self.total_chars,
            "unique_chars": unique_chars,
            "laplace_alpha": self.laplace_alpha,
            "top_10_chars": self.char_frequency.most_common(10)
        }

        return stats

    def save(self, output_path: str) -> None:
        """
        Save corpus statistics to a pickle file for later reuse.

        Serializes character frequency counts, total character count, and
        Laplace smoothing coefficient for reproducible downstream processing.

        Args:
            output_path (str): Path to output pickle file (.pkl)

        Raises:
            IOError: If output directory is not writable
        """
        # Prepare statistics for serialization
        stats = {
            "char_frequency": self.char_frequency,
            "total_chars": self.total_chars,
            "unique_chars": len(self.char_frequency),
            "laplace_alpha": self.laplace_alpha
        }

        # Save to pickle file (binary format)
        with open(output_path, 'wb') as f:
            pickle.dump(stats, f)

        self.logger.info(f"Corpus statistics saved to: {output_path}")

    def load(self, stats_path: str) -> None:
        """
        Load precomputed corpus statistics from a pickle file.

        Restores character frequency counts, total character count, and
        Laplace smoothing coefficient for reuse in processing pipelines.

        Args:
            stats_path (str): Path to input pickle file (.pkl)

        Raises:
            FileNotFoundError: If statistics file does not exist
            pickle.UnpicklingError: If file is not a valid pickle file
            KeyError: If required statistics keys are missing
        """
        # Load from pickle file
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # Restore statistics with fallback values
        self.char_frequency = stats["char_frequency"]
        self.total_chars = stats["total_chars"]
        self.laplace_alpha = stats.get("laplace_alpha", 1.0)

        self.logger.info(f"Corpus statistics loaded: {len(self.char_frequency)} unique characters")


if __name__ == "__main__":
    # Example usage: Process training corpus and save statistics
    cs = CorpusStatistics()
    stats = cs.process_directory("Data/train")
    cs.save("Output/train/run1/corpus_stats.pkl")

    # Log example statistics for verification
    print(f"CAM-15 Corpus Statistics (v1.0.0)")
    print(f"Author: ChaoJiht666")
    print(f"Total characters processed: {stats['total_chars']}")
    print(f"Unique characters: {stats['unique_chars']}")
    print(f"Recommended Laplace alpha: {stats['laplace_alpha']:.2f}")
    print(f"Top 10 characters: {stats['top_10_chars']}")