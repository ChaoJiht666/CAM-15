#!/usr/bin/env python3
"""
CAM-15/35 Interactive Prediction Script (Confidence Calibration Version)
Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Confidence calibration to fix overfitting-induced false high confidence scores
- Whitebox decision rule explanation with uncertainty quantification
- Multiple prediction modes: interactive, single text, batch file processing
- Compressed matrix format support (.joblib.xz, .joblib, .npz)
- Comprehensive reliability assessment with warning system
- OOD (Out-of-Distribution) detection based on training sample support
- Beautiful formatted output with color-coded reliability indicators

Key Fixes:
- Correct parameter name usage in confidence calibration
- Robust model deserialization with complete WhiteBoxTree class definition
- Consistent text preprocessing matching training pipeline
- Feature dimension validation and error handling
"""

import sys
import os
import string
import json
import argparse
import lzma
import math
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# CAM module imports with error handling
try:
    from src.main import CAM4Operator
    from src.Feature_Sequence_Output import FeatureSequenceOutput
except ImportError as e:
    print(f"❌ Error: Cannot import project modules: {e}")
    print("Please ensure the script is in the project's scripts/ directory or set PROJECT_ROOT manually")
    sys.exit(1)


# ============================================
# Critical: WhiteBoxTree class for deserialization
# Must match exactly with the definition in train.py
# ============================================
class WhiteBoxTree:
    """
    Whitebox decision tree wrapper with rule extraction and explainability.

    Critical Note: This class definition must remain identical to train.py
    to ensure proper joblib deserialization of saved models.

    Enhanced Features:
        - Class distribution tracking in leaf nodes
        - Entropy calculation for uncertainty quantification
        - Complete rule path extraction with statistical information
    """

    def __init__(self, tree, feature_names, class_names):
        self.tree = tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.rules = self._extract_rules()

    def _extract_rules(self):
        """Extract decision rules with complete statistical information."""
        tree_ = self.tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        rules = []

        def recurse(node, path):
            # Internal node (split condition)
            if tree_.feature[node] != -2:
                name = self.feature_names[feature[node]]
                th = threshold[node]
                left_path = path + [f"{name}<={th:.3f}"]
                recurse(tree_.children_left[node], left_path)
                right_path = path + [f"{name}>{th:.3f}"]
                recurse(tree_.children_right[node], right_path)
            # Leaf node (prediction)
            else:
                class_dist = tree_.value[node][0]
                pred = self.class_names[np.argmax(class_dist)]
                samples = int(np.sum(class_dist))
                prob = float(np.max(class_dist)) / (samples + 1e-8)
                rules.append({
                    'conditions': path,
                    'prediction': str(pred),
                    'probability': float(prob),
                    'samples': int(samples),
                    'class_dist': class_dist.tolist()
                })

        recurse(0, [])
        return rules

    def predict(self, X):
        """Proxy method for original tree prediction."""
        return self.tree.predict(X)

    def predict_proba(self, X):
        """Proxy method for original tree probability prediction."""
        return self.tree.predict_proba(X)

    def score(self, X, y):
        """Proxy method for original tree accuracy score."""
        return self.tree.score(X, y)

    def explain(self, x):
        """
        Explain prediction for single input with comprehensive statistics.

        Args:
            x: Single input feature vector (1D array/list)

        Returns:
            dict: Comprehensive explanation with rule path, confidence, entropy,
                  sample counts, and class distribution
        """
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        leaf_id = self.tree.apply(x)[0]
        tree_ = self.tree.tree_

        # Trace decision path
        path_desc = []
        node = 0
        while node != leaf_id:
            feat_idx = tree_.feature[node]
            th = tree_.threshold[node]
            if x[0][feat_idx] <= th:
                path_desc.append(f"{self.feature_names[feat_idx]}<={th:.3f}")
                node = tree_.children_left[node]
            else:
                path_desc.append(f"{self.feature_names[feat_idx]}>{th:.3f}")
                node = tree_.children_right[node]

        # Leaf node statistics
        class_dist = tree_.value[leaf_id][0]
        pred = self.class_names[np.argmax(class_dist)]
        total_samples = int(np.sum(class_dist))
        max_samples = int(np.max(class_dist))

        # Raw confidence (purity of leaf node)
        raw_conf = float(max_samples) / (total_samples + 1e-8)

        # Entropy calculation (uncertainty measure)
        probs = class_dist / (total_samples + 1e-8)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))

        return {
            'rule': " AND ".join(path_desc),
            'prediction': str(pred),
            'raw_confidence': float(raw_conf),
            'matched_samples': total_samples,
            'max_class_samples': max_samples,
            'entropy': float(entropy),
            'class_distribution': class_dist.tolist(),
            'leaf_id': int(leaf_id)
        }


def clean_text(text: str) -> str:
    """
    Clean text by removing all punctuation (Chinese + English) - identical to training.

    Critical Note: Must maintain exact consistency with training preprocessing
    to avoid feature distribution mismatch.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text with no punctuation characters
    """
    if not isinstance(text, str):
        text = str(text)

    # Define punctuation sets (Chinese + English)
    punctuation_chars = set(string.punctuation)
    chinese_punctuation = {'，', '。', '！', '？', '；', '：', '“', '”', '‘', '’',
                           '（', '）', '【', '】', '《', '》', '…', '—', '·', '、'}
    all_punctuation = punctuation_chars.union(chinese_punctuation)

    # Remove punctuation characters
    cleaned = ''.join([char for char in text if char not in all_punctuation])
    return cleaned


def find_matrix_file(matrix_dir: str, level: str = "single") -> str:
    """
    Smart matrix file detection with compressed format support.

    Priority Order (smallest to largest file size):
        1. .joblib.xz (LZMA compressed - recommended)
        2. .joblib (uncompressed joblib)
        3. .npz (standard scipy sparse format)
        4. meta.json → inferred compressed path

    Fallback Logic (hierarchical mode):
        - phrase → word level if phrase matrix not found
        - sentence → word level if sentence matrix not found

    Args:
        matrix_dir (str): Directory containing matrix files
        level (str): Matrix level (single/word/phrase/sentence)

    Returns:
        str: Path to detected matrix file

    Raises:
        FileNotFoundError: If no matrix file found in any supported format
    """
    matrix_path = Path(matrix_dir)

    # Determine base filename based on level
    if level == "single":
        base_name = "cooccur_matrix"
    else:
        base_name = f"cooccur_matrix_{level}"

    # Priority 1: LZMA compressed joblib (smallest size)
    lzma_file = matrix_path / f"{base_name}_matrix.joblib.xz"
    if lzma_file.exists():
        return str(lzma_file)

    # Priority 2: Uncompressed joblib
    joblib_file = matrix_path / f"{base_name}_matrix.joblib"
    if joblib_file.exists():
        return str(joblib_file)

    # Priority 3: Meta.json → inferred compressed path
    meta_file = matrix_path / f"{base_name}_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            compressed = meta.get('compressed_path')
            if compressed and Path(compressed).exists():
                return compressed
        except Exception:
            pass

    # Priority 4: Standard NPZ format (backward compatibility)
    npz_file = matrix_path / f"{base_name}.npz"
    if npz_file.exists():
        return str(npz_file)

    # Hierarchical mode fallback logic
    if level in ['phrase', 'sentence'] and level != 'word':
        print(f"  ⚠️  Warning: {level} level matrix not found, falling back to word level")
        return find_matrix_file(matrix_dir, "word")

    # If no files found, raise error
    raise FileNotFoundError(
        f"Could not find {level} level matrix file in {matrix_dir}."
    )


class CAMPredictor:
    """
    CAM Predictor with confidence calibration and OOD detection.

    Core Features:
        - Model loading with compressed matrix support
        - Confidence calibration to fix false high confidence scores
        - Whitebox rule explanation with uncertainty quantification
        - Multiple prediction modes (single, batch, interactive)
        - Comprehensive reliability assessment
    """

    def __init__(self, train_run: str):
        """
        Initialize CAM predictor.

        Args:
            train_run: Training directory name (e.g., 'run31') or full path

        Raises:
            FileNotFoundError: If training directory does not exist
            RuntimeError: If model loading fails
        """
        # Determine training directory path
        if os.path.exists(train_run) and os.path.isdir(train_run):
            self.train_dir = Path(train_run)
        else:
            self.train_dir = Path(PROJECT_ROOT) / 'Output' / 'train' / train_run

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        # Initialize model components
        self.clf = None
        self.scaler = None
        self.le = None
        self.extractor = None
        self.config = {}
        self.expected_dim = 15
        self.use_layered = False
        self.n_classes = 0
        self.feature_names = []

        # Load model components
        self._load_model()

    def _load_model(self):
        """Load all model components with comprehensive error handling."""
        print(f"🔄 Loading model from: {self.train_dir}")

        # Load training metadata
        meta_path = self.train_dir / 'classifier_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            self.expected_dim = self.config.get('feature_dim', 15)
            self.use_layered = self.config.get('use_layered', False)
            self.feature_names = self.config.get(
                'feature_names',
                [f"Feat_{i}" for i in range(self.expected_dim)]
            )

            print(f"  ✅ Configuration loaded successfully")
            print(f"  📊 Model Type: {self.config.get('model_type', 'WhiteBox Tree')}")
            print(f"  📏 Feature Dimension: {self.expected_dim}")
            print(f"  🏗️  Matrix Format: {'Hierarchical' if self.use_layered else 'Single Layer'}")
        else:
            print("  ⚠️  Warning: Metadata not found, using default configuration")
            self.feature_names = [f"Feat_{i}" for i in range(self.expected_dim)]

        # Load core model components
        try:
            self.clf = joblib.load(self.train_dir / 'whitebox_tree.pkl')
            self.scaler = joblib.load(self.train_dir / 'scaler.pkl')
            self.le = joblib.load(self.train_dir / 'label_encoder.pkl')
            self.n_classes = len(self.le.classes_)

            print(f"  ✅ Classifier loaded successfully")
            print(f"  📚 Number of Classes: {self.n_classes}")
            print(f"  📋 Classes: {', '.join(self.le.classes_)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}")

        # Load matrix and initialize feature extractor
        try:
            if self.use_layered:
                # Hierarchical CAM-35 mode
                matrix_file_word = find_matrix_file(str(self.train_dir), "word")
                print(f"  ✅ Hierarchical matrix loaded: Word={Path(matrix_file_word).name}")
                matrix_path_for_extractor = str(self.train_dir)
                cooccur_path_for_extractor = None
            else:
                # Single-layer CAM-15 mode
                matrix_file = find_matrix_file(str(self.train_dir), "single")
                print(f"  ✅ Single-layer matrix loaded: {Path(matrix_file).name}")
                matrix_path_for_extractor = None
                cooccur_path_for_extractor = matrix_file

            # Vocabulary path
            vocab_path = self.train_dir / 'vocab.pkl'

            # Configuration file handling
            config_path = self.train_dir / 'train_config.json'
            if not config_path.exists():
                config_path = Path(PROJECT_ROOT) / 'Config/System_Config.json'

            # Initialize CAM operator (for configuration loading)
            _ = CAM4Operator(str(config_path))

            # Initialize feature extractor
            self.extractor = FeatureSequenceOutput(
                vocab_path=str(vocab_path),
                cooccur_matrix_path=cooccur_path_for_extractor,
                matrix_dir=matrix_path_for_extractor,
                config_path=str(config_path),
                use_layered=self.use_layered
            )
            print(f"  ✅ Feature extractor initialized successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize feature extractor: {e}")

        print("✅ Model loading complete!\n")

    def calibrate_confidence(self, raw_confidence: float, matched_samples: int,
                             entropy: float, rule_depth: int) -> Tuple[float, str, List[str]]:
        """
        Calibrate raw confidence score to fix overfitting-induced false high values.

        Calibration Strategy:
            1. Laplace smoothing to avoid 100% confidence scores
            2. Sample count penalty (small samples = higher uncertainty)
            3. Rule complexity penalty (long paths = potential overfitting)
            4. Entropy-based uncertainty adjustment

        Args:
            raw_confidence: Raw confidence from decision tree (0-1)
            matched_samples: Number of training samples in leaf node
            entropy: Entropy of class distribution in leaf node (0 = pure, log2(n_classes) = max)
            rule_depth: Number of conditions in decision rule

        Returns:
            tuple: (calibrated_confidence, reliability_level, warnings)
        """
        warnings = []

        # 1. Laplace smoothing (Bayesian posterior)
        alpha = 1  # Smoothing parameter
        smoothed_conf = (matched_samples * raw_confidence + alpha) / (
                matched_samples + self.n_classes * alpha
        )

        # 2. Sample count penalty (statistical significance)
        if matched_samples < 5:
            sample_penalty = 0.3
            warnings.append(
                f"⚠️  This decision is based on only {matched_samples} samples (very low statistical significance)")
        elif matched_samples < 10:
            sample_penalty = 0.6
            warnings.append(f"⚠️  Few samples ({matched_samples}) - interpret with caution")
        elif matched_samples < 30:
            sample_penalty = 0.8
        else:
            sample_penalty = 1.0

        # 3. Rule complexity penalty (path length = potential overfitting)
        if rule_depth > 15:
            complexity_penalty = 0.7
            warnings.append("⚠️  Decision rule is too complex (potential overfitting)")
        elif rule_depth > 10:
            complexity_penalty = 0.85
        else:
            complexity_penalty = 1.0

        # 4. Entropy penalty (higher entropy = higher uncertainty)
        max_entropy = math.log2(self.n_classes) if self.n_classes > 1 else 1
        entropy_penalty = 1 - (entropy / (max_entropy + 1e-8)) ** 0.5

        # Combine all penalties
        calibrated = smoothed_conf * sample_penalty * complexity_penalty * entropy_penalty

        # Determine reliability level
        if calibrated > 0.8 and matched_samples >= 10:
            reliability = "High Reliability 🟢"
        elif calibrated > 0.5:
            reliability = "Medium Reliability 🟡"
        else:
            reliability = "Low Reliability 🔴"

        return calibrated, reliability, warnings

    def predict(self, text: str, show_details: bool = True) -> Dict[str, Any]:
        """
        Predict single text with confidence calibration and whitebox explanation.

        Args:
            text: Input text for prediction
            show_details: Whether to print detailed results

        Returns:
            dict: Comprehensive prediction results with confidence calibration
        """
        # Text cleaning (must match training preprocessing)
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return {
                'success': False,
                'error': 'Text is empty after cleaning',
                'text': text
            }

        try:
            # Feature extraction
            features = self.extractor.transform(cleaned_text, return_sequence=False)

            # Validate feature dimension
            if len(features) != self.expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.expected_dim}, got {len(features)}"
                )

            # Feature standardization
            features_scaled = self.scaler.transform(features.reshape(1, -1))[0]

            # Prediction with probabilities
            proba = self.clf.predict_proba(features_scaled.reshape(1, -1))[0]
            pred_idx = np.argmax(proba)
            pred_label = self.le.inverse_transform([pred_idx])[0]
            raw_confidence = float(proba[pred_idx])

            # Whitebox explanation with rule extraction
            explanation = self.clf.explain(features_scaled)
            rule_depth = len(explanation['rule'].split(' AND '))

            # Confidence calibration (fixed parameter name usage)
            calibrated_conf, reliability, warnings = self.calibrate_confidence(
                raw_confidence=explanation['raw_confidence'],
                matched_samples=explanation['matched_samples'],
                entropy=explanation['entropy'],
                rule_depth=rule_depth
            )

            # All class probabilities (sorted)
            all_proba_raw = {
                self.le.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }

            # Compile comprehensive results
            result = {
                'success': True,
                'text': text,
                'cleaned_text': cleaned_text,
                'prediction': pred_label,
                'raw_confidence': raw_confidence,  # Uncalibrated (potentially false) confidence
                'calibrated_confidence': calibrated_conf,  # Calibrated (realistic) confidence
                'reliability': reliability,
                'matched_samples': explanation['matched_samples'],
                'rule': explanation['rule'],
                'rule_depth': rule_depth,
                'entropy': explanation['entropy'],
                'warnings': warnings,
                'all_probabilities': dict(
                    sorted(all_proba_raw.items(), key=lambda x: x[1], reverse=True)
                ),
                'feature_vector': features_scaled.tolist()
            }

            # Print formatted results if requested
            if show_details:
                self._print_result(result)

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'text': text
            }

    def _print_result(self, result: Dict[str, Any]):
        """
        Print beautifully formatted prediction results with color coding.

        Features:
            - Color-coded reliability indicators
            - Warning system for low confidence/statistical significance
            - Simplified rule display (first 3 conditions)
            - Top-3 probability distribution with progress bars
            - OOD detection suggestions
        """
        print("=" * 70)

        # Print warnings first (high visibility)
        if result.get('warnings'):
            print("【⚠️  MODEL WARNINGS】")
            for warning in result['warnings']:
                print(f"  {warning}")
            print("-" * 70)

        # Input text (truncated for readability)
        input_text = result['text']
        if len(input_text) > 60:
            input_display = input_text[:60] + "..."
        else:
            input_display = input_text
        print(f"📝 Input: {input_display}")

        # Prediction result with reliability indicator
        print(f"\n🏷️  Predicted Class: {result['prediction']}")
        print(f"🎯 Reliability: {result['reliability']}")
        print(f"📊 Confidence: {result['calibrated_confidence']:.2%} "
              f"(Raw: {result['raw_confidence']:.2%})")
        print(f"📚 Training Support: {result['matched_samples']} samples | "
              f"Rule Depth: {result['rule_depth']} levels")

        # Top-3 probability distribution
        print(f"\n【Class Probability Distribution】")
        probs = list(result['all_probabilities'].items())[:3]
        for i, (cls, prob) in enumerate(probs, 1):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            marker = " <-- Prediction" if cls == result['prediction'] else ""
            print(f"  {i}. {cls:12} {prob:6.2%} [{bar}]{marker}")

        # Probability gap warning
        if len(probs) >= 2:
            gap = probs[0][1] - probs[1][1]
            if gap < 0.2:
                print(f"\n  💡 Note: Small gap ({gap:.1%}) between top two classes - high uncertainty")

        # Simplified decision rule display
        print(f"\n【Decision Rule】(Simplified)")
        conditions = result['rule'].split(' AND ')
        if len(conditions) > 3:
            for cond in conditions[:3]:
                print(f"  • {cond}")
            print(f"  ... and {len(conditions) - 3} more conditions")
        else:
            for cond in conditions:
                print(f"  • {cond}")

        # OOD detection suggestion
        if result['matched_samples'] < 5:
            print(f"\n  💡 OOD Warning: This appears to be an unseen text type - manual review recommended")

        print("=" * 70)

    def batch_predict(self, texts: List[str], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Batch prediction with progress tracking and optional CSV output.

        Args:
            texts: List of texts for prediction
            output_file: Optional path to save CSV results

        Returns:
            list: Comprehensive results for all input texts
        """
        results = []
        total_texts = len(texts)
        print(f"🚀 Starting batch prediction for {total_texts} texts...\n")

        for i, text in enumerate(texts, 1):
            # Predict single text (suppress detailed output)
            result = self.predict(text, show_details=False)
            results.append(result)

            # Progress display
            if result['success']:
                status = "✅" if result['calibrated_confidence'] > 0.5 else "⚠️"
                conf = result['calibrated_confidence']
                pred = result['prediction']
                text_preview = text[:30] + "..." if len(text) > 30 else text
                print(f"[{i}/{total_texts}] {status} {pred} ({conf:.1%}) | {text_preview}")
            else:
                error_preview = result['error'][:30] + "..." if len(result['error']) > 30 else result['error']
                print(f"[{i}/{total_texts}] ❌ Error: {error_preview}")

        # Save results to CSV if requested
        if output_file and results:
            output_data = []
            for r in results:
                if r['success']:
                    # Successful prediction entry
                    rule_preview = (r['rule'][:100] + '...') if len(r['rule']) > 100 else r['rule']
                    output_data.append({
                        'text': r['text'],
                        'prediction': r['prediction'],
                        'calibrated_confidence': r['calibrated_confidence'],
                        'raw_confidence': r['raw_confidence'],
                        'reliability': r['reliability'],
                        'matched_samples': r['matched_samples'],
                        'rule': rule_preview
                    })
                else:
                    # Error entry
                    output_data.append({
                        'text': r['text'],
                        'prediction': 'ERROR',
                        'error': r['error']
                    })

            # Save to CSV with UTF-8 encoding
            df = pd.DataFrame(output_data)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n💾 Results saved to: {output_file}")

        return results


def interactive_mode(predictor: CAMPredictor):
    """
    Interactive prediction mode with user input handling.

    Features:
        - Continuous prediction until 'quit' command
        - Batch input mode with empty line termination
        - Comprehensive error handling
    """
    print("\n" + "=" * 70)
    print("🔹 Entering Interactive Prediction Mode 🔹")
    print("   - Type 'quit'/'exit'/'q' to exit")
    print("   - Type 'batch' for multi-line input mode")
    print("   - Enter text directly for single prediction")
    print("=" * 70 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("📝 Enter text > ").strip()

            # Exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting prediction program")
                break

            # Batch input mode
            if user_input.lower() == 'batch':
                print("\n📥 Batch Input Mode (enter empty line to finish):")
                lines = []
                while True:
                    line = input("   > ")
                    if not line.strip():
                        break
                    lines.append(line)

                if lines:
                    print(f"\n📊 Processing {len(lines)} texts in batch...")
                    predictor.batch_predict(lines)
                else:
                    print("⚠️  No texts entered for batch processing")
                continue

            # Skip empty input
            if not user_input:
                continue

            # Single prediction
            result = predictor.predict(user_input)

            # Error handling
            if not result['success']:
                print(f"❌ Prediction failed: {result['error']}")
                if 'traceback' in result:
                    print(result['traceback'])

        except KeyboardInterrupt:
            print("\n\n👋 Exiting prediction program (keyboard interrupt)")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """
    Main entry point with comprehensive command-line interface.

    Supports multiple operation modes:
        1. Interactive mode (default)
        2. Single text prediction
        3. Batch file processing
    """
    # Command-line argument parser with examples
    parser = argparse.ArgumentParser(
        description='CAM-15/35 Interactive Prediction (Confidence Calibration)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python predict.py --run run41

  # Single text prediction
  python predict.py --run run41 --text "Where should I stay tonight?"

  # Batch prediction from file
  python predict.py --run run41 --file test_sentences.txt --output results.csv

  # Using full model path
  python predict.py --model-path /path/to/train/run41 --text "Test text"
        """
    )

    # Core arguments
    parser.add_argument('--run', type=str, help='Training run directory name (e.g., run41)')
    parser.add_argument('--model-path', type=str, help='Full path to training directory (alternative to --run)')
    parser.add_argument('--text', '-t', type=str, help='Single text for prediction')
    parser.add_argument('--file', '-f', type=str, help='File with texts (one per line) for batch prediction')
    parser.add_argument('--output', '-o', type=str, help='Output CSV path for batch prediction results')

    # Parse arguments
    args = parser.parse_args()

    # Determine model path
    if args.model_path:
        train_path = args.model_path
    elif args.run:
        train_path = args.run
    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --run or --model-path")
        sys.exit(1)

    # Initialize predictor with error handling
    try:
        predictor = CAMPredictor(train_path)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Execute appropriate prediction mode
    if args.text:
        # Single text prediction
        result = predictor.predict(args.text)
        if not result['success']:
            print(f"\n❌ Prediction failed: {result['error']}")
            sys.exit(1)

    elif args.file:
        # Batch prediction from file
        if not Path(args.file).exists():
            print(f"❌ Error: File not found - {args.file}")
            sys.exit(1)

        # Read texts from file (skip empty lines)
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"📄 Loaded {len(texts)} texts from file: {args.file}")
        predictor.batch_predict(texts, args.output)

    else:
        # Interactive mode (default)
        interactive_mode(predictor)


if __name__ == "__main__":
    main()