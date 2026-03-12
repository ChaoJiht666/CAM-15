#!/usr/bin/env python3
"""
CAM-15/35 Whitebox Batch Testing Script (v1.0.0)
Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Deserialization fix for WhiteBoxTree class (critical for joblib loading)
- Compressed matrix format support (.joblib.xz, .joblib, .npz with auto-detection)
- Comprehensive batch testing with whitebox rule explanation
- OOD (Out-of-Distribution) performance analysis vs in-distribution training
- Detailed error pattern analysis and confusion matrix generation
- Automatic timestamped output directory creation
- Full compatibility with hierarchical CAM-35 and single-layer CAM-15

Key Fixes:
- Complete WhiteBoxTree class redefinition (fixes joblib deserialization)
- Smart matrix file detection with fallback logic for hierarchical mode
- Numpy type conversion for JSON serialization
- Robust error handling for text processing and feature extraction
"""

import sys
import os
import string
import datetime
import traceback
import json
import lzma
from pathlib import Path
from collections import defaultdict, Counter

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# CAM module imports
from src.main import CAM4Operator
from src.Feature_Sequence_Output import FeatureSequenceOutput


# ============================================
# Critical: WhiteBoxTree class redefinition (required for joblib deserialization)
# Must be identical to the definition in train.py to ensure proper loading
# ============================================
class WhiteBoxTree:
    """
    Whitebox decision tree wrapper with rule extraction and explainability.

    Critical Note: This class definition must remain identical to train.py
    to ensure proper joblib deserialization of saved models.

    Key Features:
        - Extract human-readable decision rules from tree structure
        - Explain individual predictions with rule matching
        - Preserve all original scikit-learn tree functionality
    """

    def __init__(self, tree, feature_names, class_names):
        self.tree = tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.rules = self._extract_rules()

    def _extract_rules(self):
        """Extract decision rules via recursive tree traversal."""
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
                prob = float(np.max(class_dist)) / (samples + 1e-8)  # Avoid division by zero
                rules.append({
                    'conditions': path,
                    'prediction': str(pred),
                    'probability': float(prob),
                    'samples': int(samples)
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
        Explain prediction for single input by tracing decision path.

        Args:
            x: Single input feature vector (1D array/list)

        Returns:
            dict: Explanation with rule, prediction, confidence, and sample count
        """
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        node_indicator = self.tree.decision_path(x)
        leaf_id = self.tree.apply(x)[0]
        tree_ = self.tree.tree_

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

        class_dist = tree_.value[leaf_id][0]
        pred = self.class_names[np.argmax(class_dist)]
        conf = float(np.max(class_dist)) / (np.sum(class_dist) + 1e-8)

        return {
            'rule': " AND ".join(path_desc),
            'prediction': str(pred),
            'confidence': float(conf),
            'matched_samples': int(np.sum(class_dist))
        }


def clean_text(text):
    """
    Clean text by removing all punctuation (Chinese + English) - identical to train.py.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text with no punctuation characters

    Note: Must maintain exact consistency with training preprocessing!
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


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Critical fix: Prevents JSON serialization errors with numpy arrays/scalars
    by converting them to native Python types.

    Args:
        obj: Any object (numpy type or nested structure containing numpy types)

    Returns:
        Serializable object with only native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


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
        except Exception as e:
            print(f"  ⚠️  Warning: Error reading meta.json: {e}")

    # Priority 4: Standard NPZ format (backward compatibility)
    npz_file = matrix_path / f"{base_name}.npz"
    if npz_file.exists():
        return str(npz_file)

    # Hierarchical mode fallback logic
    if level in ['phrase', 'sentence'] and level != 'word':
        print(f"  ⚠️  Warning: {level} level matrix not found, falling back to word level")
        return find_matrix_file(matrix_dir, "word")

    # If no files found, raise error with search details
    searched = [
        f"{base_name}_matrix.joblib.xz",
        f"{base_name}_matrix.joblib",
        f"{base_name}.npz",
        f"{base_name}_meta.json"
    ]
    raise FileNotFoundError(
        f"Could not find {level} level matrix file in {matrix_dir}.\n"
        f"Searched for: {', '.join(searched)}"
    )


def setup_run_directory(base_dir: str) -> str:
    """
    Create timestamped test output directory to avoid overwriting results.

    Args:
        base_dir (str): Base directory for test outputs

    Returns:
        str: Path to newly created timestamped directory

    Format: YYYYMMDD_HHMMSS (e.g., 20240520_143025)
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"test_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    return str(run_dir)


def test_whitebox(test_path: str, train_run: str, show_details: int = 10):
    """
    Comprehensive whitebox batch testing with rule explanation and compressed format support.

    Performs:
        1. Model loading (with deserialization fix)
        2. Smart matrix file detection (compressed formats)
        3. Batch feature extraction and prediction
        4. Whitebox rule explanation for each prediction
        5. Performance analysis (accuracy, error patterns)
        6. OOD vs in-distribution performance comparison
        7. Detailed result saving (CSV, JSON, confusion matrix)

    Args:
        test_path (str): Path to test CSV file (must contain 'text' and 'title' columns)
        train_run (str): Training run directory name (e.g., "run31")
        show_details (int): Number of detailed predictions to show in console (default: 10)
    """
    # Validate training directory
    train_dir = Path(PROJECT_ROOT) / 'Output' / 'train' / train_run
    if not train_dir.exists():
        print(f"❌ Error: Training directory not found - {train_dir}")
        return

    # Create test output directory
    test_output_dir = setup_run_directory(os.path.join(PROJECT_ROOT, 'Output', 'test'))

    # Initial setup information
    print("=" * 80)
    print("CAM-15/35 Whitebox Batch Testing (Rule Explanation + Compressed Format Support)")
    print("=" * 80)
    print(f"📁 Training Directory: {train_dir}")
    print(f"📄 Test File: {test_path}")
    print(f"📊 Test Output: {test_output_dir}")
    print("-" * 80)

    # Load training metadata
    print("🔧 Loading training configuration...")
    meta_path = train_dir / 'classifier_meta.json'
    config = {}

    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Extract configuration parameters
        expected_dim = config.get('feature_dim', 15)
        use_layered = config.get('use_layered', False)
        train_acc = config.get('train_accuracy_in_distribution', 0)

        # Display training metadata
        print(f"  📌 Model Type: {config.get('model_type', 'WhiteBox Decision Tree')}")
        if isinstance(train_acc, (int, float)):
            print(f"  📈 Training Accuracy (In-Dist): {train_acc:.2%}")
        print(f"  🗜️  Compression: {config.get('compression', {}).get('algorithm', 'unknown')}")
        print(f"  ⚠️  Note: Training accuracy is upper bound - OOD performance is typically lower")
    else:
        # Fallback to default configuration
        expected_dim = 15
        use_layered = False
        print("  ⚠️  Warning: Metadata not found - using default configuration")

    # Load model components
    print("\n🧠 Loading whitebox model...")
    try:
        # Load core model components (with deserialization fix)
        clf = joblib.load(train_dir / 'whitebox_tree.pkl')
        scaler = joblib.load(train_dir / 'scaler.pkl')
        le = joblib.load(train_dir / 'label_encoder.pkl')

        # Smart matrix file detection
        if use_layered:
            # Hierarchical CAM-35 mode
            matrix_file_word = find_matrix_file(str(train_dir), "word")
            matrix_file_phrase = find_matrix_file(str(train_dir), "phrase")
            matrix_file_sentence = find_matrix_file(str(train_dir), "sentence")

            print(f"  📊 Matrix Files: Word={Path(matrix_file_word).name}, "
                  f"Phrase={Path(matrix_file_phrase).name}, "
                  f"Sentence={Path(matrix_file_sentence).name}")

            # Directory input for layered mode
            matrix_path_for_extractor = str(train_dir)
            cooccur_path_for_extractor = None
        else:
            # Single-layer CAM-15 mode
            matrix_file = find_matrix_file(str(train_dir), "single")
            print(f"  📊 Matrix File: {Path(matrix_file).name} ({Path(matrix_file).suffix} format)")

            # File path input for single mode
            matrix_path_for_extractor = None
            cooccur_path_for_extractor = matrix_file

        # Configuration file handling
        vocab_path = train_dir / 'vocab.pkl'
        config_path = train_dir / 'train_config.json'

        if not config_path.exists():
            config_path = Path(PROJECT_ROOT) / 'Config/System_Config.json'

        # Initialize CAM feature extractor
        cam4 = CAM4Operator(str(config_path))
        extractor = FeatureSequenceOutput(
            vocab_path=str(vocab_path),
            cooccur_matrix_path=cooccur_path_for_extractor,
            matrix_dir=matrix_path_for_extractor,
            config_path=str(config_path),
            use_layered=use_layered
        )

        # Model loading success message
        print("  ✅ Whitebox model loaded successfully")
        print(f"  📊 Number of Classes: {len(le.classes_)}")
        print(f"  📏 Feature Dimension: {expected_dim}")
        print(f"  🏗️  Matrix Format: {'Hierarchical' if use_layered else 'Single Layer'}")
        print(f"  🔍 Whitebox Features: IF-THEN decision rule explanation enabled")

    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        traceback.print_exc()
        return

    # Load test data
    print(f"\n📥 Loading test data: {test_path}")
    try:
        df = pd.read_csv(test_path)

        # Validate required columns
        if 'text' not in df.columns or 'title' not in df.columns:
            print("❌ Error: Test CSV must contain 'text' and 'title' columns")
            return

        print(f"  ✅ Loaded {len(df)} test samples")
    except Exception as e:
        print(f"  ❌ Failed to load test data: {e}")
        return

    # Batch prediction with whitebox explanation
    results = []
    errors = []
    y_true = []
    y_pred = []

    print(f"\n🔍 Starting prediction (showing first {show_details} detailed results)...")
    print("=" * 80)

    for idx, row in df.iterrows():
        raw_text = str(row['text'])
        true_label = str(row['title'])

        try:
            # Text cleaning (must match training preprocessing)
            cleaned_text = clean_text(raw_text)

            if not cleaned_text:
                errors.append({
                    'idx': int(idx),
                    'text': str(raw_text[:50]),
                    'error': 'Empty text after cleaning'
                })
                continue

            # Feature extraction
            features = extractor.transform(cleaned_text, return_sequence=False)

            # Validate feature dimension
            if len(features) != expected_dim:
                raise ValueError(f"Feature dimension mismatch: expected {expected_dim}, got {len(features)}")

            # Feature standardization
            features_scaled = scaler.transform(features.reshape(1, -1))[0]

            # Prediction
            pred_idx = clf.predict(features_scaled.reshape(1, -1))[0]
            pred_label = le.inverse_transform([int(pred_idx)])[0]
            proba = clf.predict_proba(features_scaled.reshape(1, -1))[0]
            confidence = float(np.max(proba))

            # Whitebox rule explanation
            explanation = clf.explain(features_scaled)

            # Result tracking
            is_correct = (pred_label == true_label)
            y_true.append(true_label)
            y_pred.append(pred_label)

            # Store detailed results
            result = {
                'idx': int(idx),
                'text': str(raw_text[:60]),
                'cleaned_text': str(cleaned_text[:40]),
                'true_label': str(true_label),
                'pred_label': str(pred_label),
                'correct': bool(is_correct),
                'confidence': float(confidence),
                'rule': str(explanation['rule']),
                'matched_samples': int(explanation['matched_samples']),
                'rule_confidence': float(explanation['confidence'])
            }
            results.append(result)

            # Display detailed results (first N samples)
            current_count = len(results)
            if current_count <= show_details:
                status = "✅" if is_correct else "❌"
                print(
                    f"\n【{current_count}】{status} Model Confidence:{confidence:.1%} | Rule Confidence:{explanation['confidence']:.1%}")
                print(f"  Text: {raw_text[:55]}...")
                print(f"  True Label: {true_label}")
                print(f"  Predicted Label: {pred_label}")
                print(f"  Whitebox Rule: IF {explanation['rule'][:70]}...")
                print(f"  Rule Basis: Based on {explanation['matched_samples']} training samples")

                if not is_correct:
                    print(
                        f"  ⚠️  Error Analysis: Model predicted '{pred_label}' via above rule, actual is '{true_label}'")
                    print(f"      Possible Causes: Overly broad rule | OOD sample | Class imbalance")

            # Progress update (every 50 samples after initial details)
            if current_count % 50 == 0 and current_count > show_details:
                acc = sum([r['correct'] for r in results]) / current_count * 100
                print(f"\n📊 Progress: {current_count}/{len(df)} | Current Accuracy: {acc:.1f}%")

        except Exception as e:
            error_msg = str(e)
            errors.append({
                'idx': int(idx),
                'text': str(raw_text[:50]),
                'error': error_msg
            })

            # Show first few errors in console
            if show_details > 0 and len(errors) <= 3:
                print(f"  ⚠️  Error processing sample {idx}: {error_msg[:100]}")
            continue

    # Performance statistics
    total = len(results)
    correct = sum([r['correct'] for r in results])
    accuracy = correct / total * 100 if total > 0 else 0

    # Test results summary
    print("\n" + "=" * 80)
    print("📋 Test Results Summary")
    print("=" * 80)
    print(f"📊 Total Samples Processed: {total}")
    print(f"✅ Correct Predictions: {correct}")
    print(f"❌ Incorrect Predictions: {total - correct}")
    print(f"🎯 Overall Accuracy: {accuracy:.2f}%")
    print(f"⚠️  Failed to Process: {len(errors)} samples")

    # In-distribution vs OOD comparison
    train_acc = config.get('train_accuracy_in_distribution', 0)
    if isinstance(train_acc, (int, float)) and train_acc > 0:
        gap = train_acc - accuracy / 100
        print(f"\n📈 Performance Comparison:")
        print(f"  Training Accuracy (In-Dist): {train_acc:.2%}")
        print(f"  Test Accuracy (OOD): {accuracy:.2f}%")
        print(f"  Performance Gap: {gap:.1%}")

        # Domain shift warning
        if gap > 0.3:
            print(f"  ⚠️  Severe Domain Shift Detected! (Gap > 30%)")
        elif gap > 0.15:
            print(f"  ⚠️  Moderate Domain Shift (Gap > 15%)")
        else:
            print(f"  ✅ Good Generalization (Gap < 15%)")

    # Error samples display
    if errors:
        print(f"\n❌ Failed Samples (first 3):")
        for e in errors[:3]:
            print(f"  - Sample {e['idx']}: {e['error'][:60]}")

    # Per-class accuracy analysis
    if total > 0:
        print("\n" + "-" * 80)
        print("📈 Per-Class Accuracy Statistics:")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for r in results:
            class_total[r['true_label']] += 1
            if r['correct']:
                class_correct[r['true_label']] += 1

        # Display per-class accuracy with progress bar
        class_acc_list = []
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label] * 100 if class_total[label] > 0 else 0
            class_acc_list.append({
                'label': label,
                'correct': class_correct[label],
                'total': class_total[label],
                'accuracy': acc
            })

        # Sort by sample count (descending)
        for item in sorted(class_acc_list, key=lambda x: x['total'], reverse=True):
            bar = "█" * int(item['accuracy'] / 5)
            print(f"  {item['label'][:15]:15} {item['correct']:3}/{item['total']:3} "
                  f"({item['accuracy']:5.1f}%) {bar}")

        # Save confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
            cm_path = Path(test_output_dir) / 'confusion_matrix.csv'
            cm_df.to_csv(cm_path, encoding='utf-8-sig')
            print(f"\n📊 Confusion Matrix Saved: {cm_path}")
        except Exception as e:
            print(f"⚠️  Failed to save confusion matrix: {e}")

    # Save detailed results
    if results:
        # Convert to serializable format and save as CSV
        results_serializable = convert_to_serializable(results)
        df_results = pd.DataFrame(results_serializable)

        csv_path = Path(test_output_dir) / 'test_results.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📄 Detailed Results Saved: {csv_path}")

        # Create and save test summary
        summary = {
            'test_timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_run': str(train_run),
            'test_file': str(test_path),
            'total_samples': int(total),
            'correct': int(correct),
            'accuracy': round(float(accuracy), 2),
            'errors': int(len(errors)),
            'train_accuracy_in_dist': float(train_acc) if isinstance(train_acc, (int, float)) else 0,
            'performance_gap': float(train_acc - accuracy / 100) if isinstance(train_acc, (int, float)) else 0,
            'feature_type': str(config.get('feature_type', 'unknown')),
            'feature_dim': int(expected_dim),
            'matrix_format': 'hierarchical' if use_layered else 'single',
            'compression_algorithm': str(config.get('compression', {}).get('algorithm', 'none'))
        }

        # Save summary as JSON
        summary = convert_to_serializable(summary)
        summary_path = Path(test_output_dir) / 'test_summary.json'

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📊 Test Summary Saved: {summary_path}")

    # Error pattern analysis
    if total > 0:
        wrong_cases = [r for r in results if not r['correct']]

        if wrong_cases:
            # Most common error patterns
            error_pairs = Counter([(r['true_label'], r['pred_label']) for r in wrong_cases])
            print(f"\n🔍 Common Error Patterns (True → Predicted):")

            for (true_l, pred_l), count in error_pairs.most_common(5):
                print(f"  {true_l} → {pred_l}: {count} occurrences")

            # Error rule analysis
            print(f"\n🔍 Error Rule Analysis (first 3 incorrect predictions):")
            for i, err in enumerate(wrong_cases[:3], 1):
                print(f"  {i}. True:{err['true_label']} → Predicted:{err['pred_label']}")
                print(f"     Rule: IF {err['rule'][:60]}...")
                print(f"     Rule Basis: {err['matched_samples']} training samples")

    print("\n" + "=" * 80)
    print("✅ Testing completed successfully!")
    print(f"📁 All results saved to: {test_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(
        description='CAM WhiteBox Batch Testing with Rule Explanation (Compressed Format Support)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file',
                        required=True,
                        help='Path to test CSV file (must contain "text" and "title" columns)')
    parser.add_argument('--run',
                        required=True,
                        help='Training run directory name (e.g., "run31")')

    # Optional argument
    parser.add_argument('--show',
                        type=int,
                        default=10,
                        help='Number of detailed predictions to show in console (default: 10, 0 for none)')

    # Parse arguments and run test
    args = parser.parse_args()
    test_whitebox(args.test_file, args.run, args.show)