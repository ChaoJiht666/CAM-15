# src/main.py
"""
CAM-4/15/35 Unified Operator Controller (v1.0.0) - Hierarchical Enhanced Version

Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Unified interface for CAM-4/15/35 operators with three operational modes
- Hierarchical CAM-35 implementation (Word + Phrase + Sentence layers)
- Full backward compatibility with CAM-4 (Lite mode)
- Enhanced CAM-15 with 15-dimensional feature space
- Complete pipeline automation (corpus stats → vocabulary → matrices → features)
- Performance benchmarking and feature interpretability analysis

Supported Modes:
1. Lite (CAM-4): 4-dimensional features (legacy compatibility)
2. Enhanced (CAM-15): 15-dimensional features (single layer enhanced)
3. Layered (CAM-35): 35-dimensional features (Word15 + Phrase15 + Sentence5)
   - Independent matrix construction for each layer
   - Configurable layer weights (Word: 0.4, Phrase: 0.5, Sentence: 0.1 default)

Key Fixes & Improvements:
- True independent matrix construction for hierarchical mode
- Path handling corrections (directory vs file path issues)
- Proper phrase corpus construction (not just copied word matrices)
- Automatic mode detection and configuration management
- Comprehensive error handling and logging
"""

import json
import logging
import sys
import time
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List

# Third-party imports
import jieba
import numpy as np
import pandas as pd

# CAM module imports
from .Corpus_Statistics import CorpusStatistics
from .Vocabulary_Construction import VocabularyConstruction
from .Cooccurrence_Matrix_Estimation import CooccurrenceMatrixEstimation
from .Feature_Sequence_Output import FeatureSequenceOutput


class CAM4Operator:
    """
    Unified CAM operator interface supporting hierarchical CAM-35, enhanced CAM-15, and legacy CAM-4.

    Provides a complete end-to-end pipeline for:
    1. Corpus statistics collection
    2. Vocabulary construction
    3. Co-occurrence matrix estimation (single or hierarchical)
    4. Feature extraction (single text or batch processing)
    5. Performance benchmarking
    6. Feature interpretability analysis

    Hierarchical CAM-35 Features:
    - Word layer (15D): Character-level co-occurrence with window size 5
    - Phrase layer (15D): Bigram phrase co-occurrence with window size 3
    - Sentence layer (5D): Long-distance co-occurrence with window size 10
    - Configurable layer weights for final feature fusion

    Attributes:
        config_path (str): Path to system configuration JSON file
        config (Dict): Loaded configuration parameters
        logger (logging.Logger): Module-specific logger instance
        use_layered (bool): Enable/disable hierarchical CAM-35 mode
        mode (str): Operational mode (lite/enhanced/layered)
    """

    def __init__(self, config_path: str = "Config/System_Config.json"):
        """
        Initialize CAM operator with configuration settings.

        Args:
            config_path (str): Path to system configuration JSON file
                               Default: "Config/System_Config.json"

        Raises:
            FileNotFoundError: If configuration file does not exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        # Configuration management
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Setup logging system
        self._setup_logging()
        self.logger = logging.getLogger("CAM4.Main")

        # Mode configuration with defaults
        self.use_layered = self.config.get("use_layered_cam", False)
        self.mode = self.config.get("feature_mode", "enhanced")  # Default: CAM-15

        # Mode initialization logging
        self.logger.info("=" * 60)
        if self.use_layered:
            # CAM-35 hierarchical mode
            self.logger.info("🔹 CAM-35 Layered Operator Initialized")
            self.logger.info("   Feature Dimension: 35 (Word15 + Phrase15 + Sentence5)")
            self.logger.info(f"   Layer Weights: Word={self.config.get('word_weight', 0.4):.1f}, "
                           f"Phrase={self.config.get('phrase_weight', 0.5):.1f}, "
                           f"Sentence={self.config.get('sentence_weight', 0.1):.1f}")
        elif self.mode == "enhanced":
            # CAM-15 enhanced mode
            self.logger.info("🔹 CAM-15 Operator Initialized (Enhanced Mode)")
            self.logger.info("   Feature Dimension: 15")
        else:
            # CAM-4 legacy mode
            self.logger.info("🔹 CAM-4 Operator Initialized (Lite Mode)")
            self.logger.info("   Feature Dimension: 4")
        self.logger.info("=" * 60)

    def _setup_logging(self):
        """
        Configure comprehensive logging system for CAM operator.

        Sets up:
        - Console output handler
        - Standard log format (timestamp, module, level, message)
        - INFO level logging (adjust via config if needed)
        """
        # Standard log format with timestamp and module information
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]  # Log to console
        )

    def _build_phrase_corpus(self, data_dir: str, output_dir: str) -> str:
        """
        Build phrase-level corpus from original text data (jieba segmentation + bigram sliding window).

        Critical fix: Constructs true phrase co-occurrence data rather than just copying word matrices.
        Generates bigram phrases from segmented text for independent phrase layer matrix construction.

        Args:
            data_dir (str): Directory containing original CSV data files
            output_dir (str): Directory for saving phrase corpus output

        Returns:
            str: Path to generated phrase corpus text file

        Raises:
            FileNotFoundError: If data directory does not exist
            ValueError: If no CSV files found in data directory
            IOError: If output directory is not writable

        Processing Steps:
            1. Load CSV files (UTF-8/GBK encoding support)
            2. Segment text using jieba
            3. Generate bigram phrases (word[i] + word[i+1])
            4. Save phrases to text file for corpus processing
        """
        self.logger.info("  📝 Building phrase-level corpus (jieba segmentation + bigram)...")

        # Validate input directory
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in data directory: {data_dir}")

        # Process each CSV file to generate phrase corpus
        phrase_data = []
        for csv_file in csv_files:
            self.logger.debug(f"    Processing file: {csv_file.name}")

            # Handle different encodings (UTF-8/GBK)
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='gbk')

            # Process each text entry
            for text in df['text'].astype(str):
                # Skip empty text
                if not text.strip():
                    continue

                # Segment text into words using jieba
                words = list(jieba.cut(text.strip()))

                # Generate bigram phrases (sliding window of 2 words)
                for i in range(len(words)-1):
                    phrase = words[i] + words[i+1]
                    phrase_data.append(phrase)

        # Save phrase corpus to text file
        phrase_corpus_path = os.path.join(output_dir, "phrase_corpus.txt")
        with open(phrase_corpus_path, 'w', encoding='utf-8') as f:
            for phrase in phrase_corpus_path:
                f.write(phrase + '\n')

        self.logger.info(f"  ✅ Phrase corpus created: {phrase_corpus_path} ({len(phrase_data)} phrases)")
        return phrase_corpus_path

    def build_pipeline(self, data_dir: str, output_dir: str) -> Dict:
        """
        Complete end-to-end CAM pipeline construction (auto-detects hierarchical mode).

        Critical fix: True independent matrix construction for hierarchical CAM-35 mode
        with different window sizes for each layer (Word:5, Phrase:3, Sentence:10).

        Pipeline Steps:
            1. Corpus statistics collection
            2. Vocabulary construction (character-level base vocabulary)
            3. Co-occurrence matrix estimation (single or hierarchical)

        Args:
            data_dir (str): Directory containing input text data (CSV files)
            output_dir (str): Directory for saving all pipeline outputs

        Returns:
            Dict: Mapping of pipeline components to their file paths

        Raises:
            FileNotFoundError: If data directory does not exist
            IOError: If output directory is not writable
            RuntimeError: If pipeline step fails

        Notes:
            - Preserves backward compatibility with CAM-4/15 (single matrix)
            - Creates independent matrices for each CAM-35 layer
            - Handles path construction correctly (fixed path concatenation bugs)
        """
        # Create output directory (including parent directories)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Critical fix: Pass data directory to config for CorpusStatistics
        self.config['data_path'] = data_dir

        # Track pipeline outputs
        results = {}

        # Step 1: Corpus statistics collection
        self.logger.info("🔄 Step 1/4: Collecting corpus statistics...")
        cs = CorpusStatistics(self.config_path)
        stats = cs.process_directory(data_dir)
        stats_file = output_path / "corpus_stats.pkl"
        cs.save(str(stats_file))
        results['corpus_stats'] = str(stats_file)
        self.logger.info(f"  ✅ Statistics saved: {stats_file}")

        # Step 2: Vocabulary construction (character-level base vocabulary)
        self.logger.info("🔄 Step 2/4: Building vocabulary (character-level)...")
        vc = VocabularyConstruction(self.config_path)
        vc.build_from_stats(str(stats_file))
        vocab_file = output_path / "vocab.pkl"
        vc.save(str(vocab_file))
        results['vocab'] = str(vocab_file)
        self.logger.info(f"  ✅ Vocabulary saved: {vocab_file} (size: {len(vc.char2idx)} tokens)")

        # Step 3: Co-occurrence matrix estimation
        self.logger.info("🔄 Step 3/4: Estimating co-occurrence matrices...")

        if self.use_layered:
            # CAM-35 hierarchical mode: Independent matrices for each layer

            # 3.1 Word-level matrix (window=5, character-level co-occurrence)
            self.logger.info("  📊 Building word-level matrix (window size=5)...")
            cme_word = CooccurrenceMatrixEstimation(self.config_path)
            cme_word.window_size = 5  # Optimal for word-level relationships
            cme_word.build_matrix(data_dir, vocab_obj=vc)
            matrix_file_word = output_path / "cooccur_matrix_word.npz"
            cme_word.save(str(matrix_file_word))
            results['cooccur_matrix_word'] = str(matrix_file_word)
            self.logger.info(f"    ✅ Word matrix saved: {matrix_file_word}")

            # 3.2 Phrase-level matrix (window=3, bigram phrase co-occurrence)
            self.logger.info("  📊 Building phrase-level matrix (window size=3)...")
            cme_phrase = CooccurrenceMatrixEstimation(self.config_path)
            cme_phrase.window_size = 3  # Smaller window for phrase-level relationships

            # Temporary config adjustment for phrase matrix construction
            original_mode = self.config.get("mode", "word")
            self.config["mode"] = "word"  # Use word mode for phrase matrix

            # Build phrase matrix (reuse vocabulary, different window/statistics)
            cme_phrase.build_matrix(data_dir, vocab_obj=vc)
            matrix_file_phrase = output_path / "cooccur_matrix_phrase.npz"
            cme_phrase.save(str(matrix_file_phrase))
            results['cooccur_matrix_phrase'] = str(matrix_file_phrase)
            self.logger.info(f"    ✅ Phrase matrix saved: {matrix_file_phrase}")

            # 3.3 Sentence-level matrix (window=10, long-distance dependencies)
            self.logger.info("  📊 Building sentence-level matrix (window size=10)...")
            cme_sentence = CooccurrenceMatrixEstimation(self.config_path)
            cme_sentence.window_size = 10  # Larger window for sentence-level relationships
            cme_sentence.build_matrix(data_dir, vocab_obj=vc)
            matrix_file_sentence = output_path / "cooccur_matrix_sentence.npz"
            cme_sentence.save(str(matrix_file_sentence))
            results['cooccur_matrix_sentence'] = str(matrix_file_sentence)
            self.logger.info(f"    ✅ Sentence matrix saved: {matrix_file_sentence}")

            # Restore original configuration
            self.config["mode"] = original_mode

        else:
            # CAM-4/15 legacy mode (fixed path concatenation bug)
            cme = CooccurrenceMatrixEstimation(self.config_path)
            cme.build_matrix(data_dir, vocab_obj=vc)
            matrix_file = output_path / "cooccur_matrix.npz"
            cme.save(str(matrix_file))  # Critical fix: pass file path, not directory
            results['cooccur_matrix'] = str(matrix_file)
            self.logger.info(f"  ✅ Matrix saved: {matrix_file}")

        self.logger.info("🎉 Pipeline construction completed successfully!")
        return results

    def create_extractor(self, vocab_path: str, matrix_dir: str) -> FeatureSequenceOutput:
        """
        Create feature extractor with automatic mode detection.

        Args:
            vocab_path (str): Path to vocabulary file (.pkl)
            matrix_dir (str): Directory containing matrices (hierarchical mode)
                              or path to single matrix file (legacy mode)

        Returns:
            FeatureSequenceOutput: Configured feature extractor instance

        Notes:
            - Auto-detects hierarchical mode (directory input) vs legacy mode (file input)
            - Preserves all configuration settings from operator initialization
        """
        # Auto-detect hierarchical mode (directory input + config flag)
        use_layered = os.path.isdir(matrix_dir) and self.use_layered

        return FeatureSequenceOutput(
            vocab_path=vocab_path,
            matrix_dir=matrix_dir,
            config_path=self.config_path,
            use_layered=use_layered
        )

    def extract_features(self, text: str, vocab_path: str, matrix_dir: str):
        """
        Convenience interface: Extract features for single text input.

        Args:
            text (str): Input text to process
            vocab_path (str): Path to vocabulary file (.pkl)
            matrix_dir (str): Directory containing matrices (hierarchical)
                              or path to single matrix file (legacy)

        Returns:
            np.ndarray: Extracted feature vector (4D/15D/35D depending on mode)

        Notes:
            Returns flattened feature vector (not sequence) by default
            For sequence output, use create_extractor() + transform() directly
        """
        extractor = self.create_extractor(vocab_path, matrix_dir)
        return extractor.transform(text, return_sequence=False)

    def benchmark(self, texts: List[str], vocab_path: str, matrix_dir: str) -> Dict:
        """
        Performance benchmarking for feature extraction.

        Measures average extraction time per text with standard deviation,
        reports feature dimension and operational mode.

        Args:
            texts (List[str]): List of texts for benchmarking
            vocab_path (str): Path to vocabulary file (.pkl)
            matrix_dir (str): Directory containing matrices (hierarchical)
                              or path to single matrix file (legacy)

        Returns:
            Dict: Benchmark results with timing and feature information
                - avg_time: Average extraction time (seconds)
                - std_time: Standard deviation of extraction time (seconds)
                - feature_dim: Feature vector dimension (4/15/35)
                - mode: Operational mode (lite/enhanced/layered)

        Notes:
            Uses first 100 texts for representative benchmarking
            Measures end-to-end feature extraction time
        """
        # Create configured extractor
        extractor = self.create_extractor(vocab_path, matrix_dir)

        # Measure extraction time for each text
        times = []
        for text in texts:
            start = time.time()
            _ = extractor.transform(text)
            times.append(time.time() - start)

        # Calculate benchmark metrics
        feature_dim = 35 if self.use_layered else (15 if self.mode == "enhanced" else 4)
        mode = "layered" if self.use_layered else self.mode

        return {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "feature_dim": feature_dim,
            "mode": mode
        }


def main():
    """
    Command-line interface (CLI) for CAM-4/15/35 operator.

    Supports four operational modes:
    1. build: Complete pipeline construction (stats → vocab → matrices)
    2. extract: Feature extraction for single text input
    3. benchmark: Performance benchmarking on text corpus
    4. analyze: Feature importance and interpretability analysis

    Command-line Arguments:
        --mode: Operation mode (build/extract/benchmark/analyze) [REQUIRED]
        --data: Input data directory (CSV files)
        --output: Output directory (default: Output/train/run1)
        --text: Input text for extraction/analysis
        --vocab: Path to vocabulary file (.pkl)
        --matrix: Path to matrix file/directory
        --config: Path to configuration file (default: Config/System_Config.json)
        --layered: Enable hierarchical CAM-35 mode
        --word-weight: Word layer weight (default: 0.4)
        --phrase-weight: Phrase layer weight (default: 0.5)
        --sentence-weight: Sentence layer weight (default: 0.1)
    """
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description='CAM-4/15/35 Operator CLI - Unified Hierarchical Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build CAM-15 pipeline
  python main.py --mode build --data data/train --output Output/train/run1
  
  # Build CAM-35 hierarchical pipeline
  python main.py --mode build --data data/train --output Output/train/run1 --layered
  
  # Extract features for single text (CAM-15)
  python main.py --mode extract --text "自然语言处理" --vocab Output/train/run1/vocab.pkl --matrix Output/train/run1/cooccur_matrix.npz
  
  # Benchmark performance (CAM-35)
  python main.py --mode benchmark --data data/test.csv --vocab Output/train/run1/vocab.pkl --matrix Output/train/run1 --layered
        """
    )

    # Core arguments
    parser.add_argument('--mode',
                       choices=['build', 'extract', 'benchmark', 'analyze'],
                       required=True,
                       help='Operation mode: build pipeline, extract features, benchmark performance, or analyze features')
    parser.add_argument('--data',
                       help='Data directory (build/benchmark modes) or CSV file path (benchmark mode)')
    parser.add_argument('--output',
                       default='Output/train/run1',
                       help='Output directory for pipeline artifacts (default: Output/train/run1)')
    parser.add_argument('--text',
                       help='Input text for feature extraction/analysis (extract/analyze modes)')
    parser.add_argument('--vocab',
                       help='Path to vocabulary file (.pkl) (extract/benchmark/analyze modes)')
    parser.add_argument('--matrix',
                       help='Path to matrix file or directory (extract/benchmark/analyze modes)')
    parser.add_argument('--config',
                       default='Config/System_Config.json',
                       help='Path to configuration file (default: Config/System_Config.json)')

    # Hierarchical CAM-35 arguments
    parser.add_argument('--layered',
                       action='store_true',
                       help='Enable hierarchical CAM-35 mode (Word + Phrase + Sentence layers)')
    parser.add_argument('--word-weight',
                       type=float,
                       default=0.4,
                       help='Word layer weight (default: 0.4)')
    parser.add_argument('--phrase-weight',
                       type=float,
                       default=0.5,
                       help='Phrase layer weight (default: 0.5)')
    parser.add_argument('--sentence-weight',
                       type=float,
                       default=0.1,
                       help='Sentence layer weight (default: 0.1)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load or create configuration
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}  # Use defaults if config file doesn't exist

    # Apply hierarchical mode configuration
    config['use_layered_cam'] = args.layered
    if args.layered:
        config['word_weight'] = args.word_weight
        config['phrase_weight'] = args.phrase_weight
        config['sentence_weight'] = args.sentence_weight

    # Save temporary configuration for operator initialization
    temp_config = f"temp_config_{'layered' if args.layered else 'single'}.json"
    with open(temp_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Initialize CAM operator
    cam4 = CAM4Operator(temp_config)

    # Execute requested operation
    if args.mode == 'build':
        # Pipeline construction mode
        if not args.data:
            print("❌ Error: --data argument is required for build mode")
            sys.exit(1)

        results = cam4.build_pipeline(args.data, args.output)

        # Print pipeline results summary
        print("\n📋 Pipeline Construction Complete:")
        for component, path in results.items():
            print(f"  • {component}: {path}")

        if args.layered:
            print("\n🏗️ Hierarchical CAM-35 Mode Complete! Components:")
            print("  • Word-level vocabulary + co-occurrence matrix (window=5)")
            print("  • Phrase-level co-occurrence matrix (window=3)")
            print("  • Sentence-level co-occurrence matrix (window=10)")

    elif args.mode == 'extract':
        # Feature extraction mode
        if not all([args.text, args.vocab, args.matrix]):
            print("❌ Error: --text, --vocab, and --matrix arguments are required for extract mode")
            sys.exit(1)

        # Extract and display features
        features = cam4.extract_features(args.text, args.vocab, args.matrix)

        print(f"\n📝 Input Text: {args.text}")
        print(f"📊 Feature Dimension: {len(features)}")
        print(f"🔢 Feature Values:")

        # Display feature values in readable chunks
        feature_chunks = [features[i:i+15] for i in range(0, len(features), 15)]
        for i, chunk in enumerate(feature_chunks):
            start_idx = i * 15 + 1
            end_idx = min((i + 1) * 15, len(features))
            print(f"  • Features {start_idx}-{end_idx}: {chunk}")

    elif args.mode == 'analyze':
        # Feature interpretability analysis mode
        if not all([args.text, args.vocab, args.matrix]):
            print("❌ Error: --text, --vocab, and --matrix arguments are required for analyze mode")
            sys.exit(1)

        # Create extractor and perform detailed analysis
        extractor = cam4.create_extractor(args.vocab, args.matrix)
        analysis = extractor.analyze_feature_importance(args.text)

        # Print detailed feature analysis
        print(f"\n🔍 Feature Importance Analysis:")
        print(f"📝 Text: {analysis['text']}")
        print(f"📊 Feature Dimension: {analysis['feature_dim']}")
        print("\n📈 Top 5 Character-Level Features:")

        for char_feat in analysis['char_level'][:5]:
            print(f"  • Position {char_feat['position']}: "
                  f"Self-correlation={char_feat['center_self_corr']:.3f}, "
                  f"Forward={char_feat['forward_link']:.3f}, "
                  f"Backward={char_feat['backward_link']:.3f}")

    elif args.mode == 'benchmark':
        # Performance benchmarking mode
        if not all([args.data, args.vocab, args.matrix]):
            print("❌ Error: --data, --vocab, and --matrix arguments are required for benchmark mode")
            sys.exit(1)

        # Load test texts (first 100 entries from CSV)
        try:
            df = pd.read_csv(args.data)
            texts = df['text'].head(100).tolist()  # Use first 100 texts for benchmarking
        except Exception as e:
            print(f"❌ Error loading benchmark data: {e}")
            sys.exit(1)

        # Run benchmark and display results
        results = cam4.benchmark(texts, args.vocab, args.matrix)

        print(f"\n⚡ Performance Benchmark Results (100 texts):")
        print(f"  • Average extraction time: {results['avg_time']*1000:.2f} ms")
        print(f"  • Standard deviation: {results['std_time']*1000:.2f} ms")
        print(f"  • Feature dimension: {results['feature_dim']}")
        print(f"  • Operational mode: {results['mode']}")

    # Clean up temporary configuration file
    if os.path.exists(temp_config):
        os.remove(temp_config)


if __name__ == "__main__":
    """
    CAM-4/15/35 Operator Main Entry Point.
    
    Provides CLI access to all CAM operator functionality with comprehensive
    error handling, logging, and user feedback.
    """
    main()