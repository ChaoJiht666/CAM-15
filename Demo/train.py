#!/usr/bin/env python3
"""
CAM-15/35 Whitebox Training Script - Full Dataset Version (v1.0.0)
Author: ChaoJiht666
Repository: GitHub (CAM-15)

Core Features:
- Full dataset training (no train/validation split) with cross-validation parameter selection
- JSON serialization fix (numpy type conversion to native Python types)
- Compressed matrix format support (.xz, .joblib, .npz with auto-detection)
- Whitebox decision tree with CCP pruning for interpretability
- Model compression with configurable LZMA levels (0-9)
- Automatic hierarchical matrix detection (CAM-35)
- Comprehensive rule extraction and feature importance analysis

Key Fixes:
- JSON serialization errors with numpy types
- Matrix path resolution (returns actual .joblib.xz instead of meta.json)
- Hierarchical matrix fallback logic (phrase/sentence → word level)
- Path handling for compressed matrix formats
"""

import sys
import os
import traceback
import string
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import json
from collections import defaultdict
import lzma  # Compression format support

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# CAM module imports
from src.main import CAM4Operator
from src.Feature_Sequence_Output import FeatureSequenceOutput


def clean_text(text):
    """
    Clean text by removing all punctuation (Chinese + English).

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text with no punctuation characters

    Notes:
        Handles both string and non-string inputs (converts to string first)
        Preserves all other characters (Chinese/English letters, numbers, spaces)
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        text = str(text)

    # Define punctuation sets
    punctuation_chars = set(string.punctuation)
    chinese_punctuation = {'，', '。', '！', '？', '；', '：', '“', '”', '‘', '’',
                           '（', '）', '【', '】', '《', '》', '…', '—', '·', '、'}
    all_punctuation = punctuation_chars.union(chinese_punctuation)

    # Remove punctuation characters
    cleaned = ''.join([char for char in text if char not in all_punctuation])
    return cleaned


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types (JSON serialization fix).

    Critical fix: Resolves JSON serialization errors with numpy arrays/scalars
    by converting them to native Python lists/floats/ints.

    Args:
        obj: Any object (numpy type or nested structure containing numpy types)

    Returns:
        Serializable object with only native Python types

    Conversion Rules:
        - np.integer → int
        - np.floating → float
        - np.ndarray → list
        - dict → recursively convert values
        - list → recursively convert elements
        - all others → return as-is
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


class WhiteBoxTree:
    """
    Whitebox decision tree wrapper with rule extraction and explainability.

    Provides human-readable decision rules and prediction explanations
    for standard scikit-learn DecisionTreeClassifier objects.

    Key Features:
        - Extract decision rules in natural language format
        - Explain individual predictions with rule matching
        - Preserve all original tree functionality (predict, predict_proba, score)
        - Serialize rules to JSON-compatible format

    Attributes:
        tree: Original scikit-learn DecisionTreeClassifier
        feature_names: List of feature names for rule explanation
        class_names: List of class names for prediction labels
        rules: Extracted decision rules (list of dictionaries)
    """

    def __init__(self, tree, feature_names, class_names):
        """
        Initialize whitebox tree wrapper.

        Args:
            tree: scikit-learn DecisionTreeClassifier instance
            feature_names (list): List of feature names (in order)
            class_names (list): List of class names (in label order)
        """
        self.tree = tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.rules = self._extract_rules()

    def _extract_rules(self):
        """
        Extract decision rules from tree structure (recursive traversal).

        Returns:
            list: Decision rules with conditions, predictions, probabilities, and sample counts

        Rule Structure:
            {
                'conditions': list of condition strings (e.g., ["Feat_0<=0.5", "Feat_3>0.2"]),
                'prediction': predicted class name,
                'probability': confidence score (0-1),
                'samples': number of training samples in this leaf node
            }
        """
        tree_ = self.tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        rules = []

        def recurse(node, path):
            # Internal node (split condition)
            if tree_.feature[node] != -2:
                name = self.feature_names[feature[node]]
                th = threshold[node]
                # Recurse on left branch (condition <= threshold)
                left_path = path + [f"{name}<={th:.3f}"]
                recurse(tree_.children_left[node], left_path)
                # Recurse on right branch (condition > threshold)
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

        # Start recursion from root node (0) with empty path
        recurse(0, [])
        return rules

    def predict(self, X):
        """Proxy method for original tree predict."""
        return self.tree.predict(X)

    def predict_proba(self, X):
        """Proxy method for original tree predict_proba."""
        return self.tree.predict_proba(X)

    def score(self, X, y):
        """Proxy method for original tree score (accuracy)."""
        return self.tree.score(X, y)

    def explain(self, x):
        """
        Explain prediction for single input by tracing decision path.

        Args:
            x: Single input feature vector (1D array/list)

        Returns:
            dict: Explanation with rule, prediction, confidence, and sample count

        Explanation Structure:
            {
                'rule': Combined condition string (e.g., "Feat_0<=0.5 AND Feat_3>0.2"),
                'prediction': Predicted class name,
                'confidence': Confidence score (0-1),
                'matched_samples': Number of training samples in matched leaf node
            }
        """
        # Ensure input is 2D array
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Trace decision path
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

        # Get prediction details
        class_dist = tree_.value[leaf_id][0]
        pred = self.class_names[np.argmax(class_dist)]
        conf = float(np.max(class_dist)) / (np.sum(class_dist) + 1e-8)

        return {
            'rule': " AND ".join(path_desc),
            'prediction': str(pred),
            'confidence': float(conf),
            'matched_samples': int(np.sum(class_dist))
        }


def find_matrix_file(matrix_dir: str, level: str = "single") -> str:
    """
    Smart matrix file detection (supports optimized compressed formats).

    Critical fix: Returns actual .joblib.xz file path instead of meta.json
    with priority-based format selection for maximum compatibility.

    Args:
        matrix_dir (str): Directory containing matrix files
        level (str): Matrix level (single/word/phrase/sentence)

    Returns:
        str: Path to detected matrix file

    Raises:
        FileNotFoundError: If no matrix file found in any supported format

    Priority Order (smallest to largest file size):
        1. _matrix.joblib.xz (LZMA compressed - smallest)
        2. _matrix.joblib (uncompressed joblib)
        3. .npz (standard scipy sparse format)
        4. meta.json → inferred .joblib.xz path

    Fallback Logic (hierarchical mode):
        - phrase → word level if phrase matrix not found
        - sentence → word level if sentence matrix not found
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
            print(f"  ⚠️  Error reading meta.json: {e}")

        # Default to expected joblib.xz path
        expected = matrix_path / f"{base_name}_matrix.joblib.xz"
        if expected.exists():
            return str(expected)

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


def train_whitebox_full(cam4_op, vocab_path, matrix_dir, data_dir, output_dir, use_layered,
                        model_compress_level: int = 3):
    """
    Train whitebox decision tree on full dataset (no train/validation split).

    Uses cross-validation to select optimal CCP pruning parameters to prevent
    overfitting while maintaining interpretability. Supports both single-layer
    (CAM-15) and hierarchical (CAM-35) feature modes with compressed matrices.

    Args:
        cam4_op: Initialized CAM4Operator instance
        vocab_path (str): Path to vocabulary file (.pkl)
        matrix_dir (str): Directory containing matrix files
        data_dir (str): Directory with training CSV files
        output_dir (str): Directory for saving model artifacts
        use_layered (bool): Enable hierarchical CAM-35 mode
        model_compress_level (int): Joblib compression level (0-9, default=3)

    Returns:
        tuple: (WhiteBoxTree classifier, StandardScaler, LabelEncoder)

    Key Features:
        - Cross-validation for CCP pruning parameter selection
        - Full dataset training with in-distribution accuracy reporting
        - Automatic matrix format detection (compressed/uncompressed)
        - Comprehensive model serialization with compression
        - Rule extraction and whitebox explainability
    """
    print("\n[4/4] Training Whitebox Decision Tree (Full Dataset Version)...")
    print("  ⚠️  Note: Training on full dataset (no train/validation split)")
    print("  ⚠️  Avoid overinterpretation of in-distribution performance")
    print("  Selecting optimal pruning parameters via cross-validation...")

    # Matrix configuration (compressed format support)
    if use_layered:
        # Hierarchical CAM-35 mode (35D features)
        matrix_file_word = find_matrix_file(matrix_dir, "word")
        matrix_file_phrase = find_matrix_file(matrix_dir, "phrase")  # May fallback to word
        matrix_file_sentence = find_matrix_file(matrix_dir, "sentence")  # May fallback to word
        expected_dim = 35
        print(f"  Matrix files: Word={Path(matrix_file_word).name}, "
              f"Phrase={Path(matrix_file_phrase).name}, "
              f"Sentence={Path(matrix_file_sentence).name}")
    else:
        # Single-layer CAM-15 mode (15D features)
        matrix_file = find_matrix_file(matrix_dir, "single")
        expected_dim = 15
        print(f"  Matrix file: {Path(matrix_file).name} ({Path(matrix_file).suffix} format)")

    # Load full dataset (all CSV files in data directory)
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    all_texts, all_labels = [], []

    for csv_file in csv_files:
        # Handle different encodings (UTF-8/GBK)
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='gbk')

        # Remove rows with missing text/label
        df = df.dropna(subset=['text', 'title'])

        # Clean text and collect data
        all_texts.extend(df['text'].astype(str).apply(clean_text).tolist())
        all_labels.extend(df['title'].astype(str).tolist())

    print(f"  Loaded full dataset: {len(all_texts)} samples")

    # Feature extraction (full dataset)
    if use_layered:
        # Hierarchical mode (directory input for automatic layer detection)
        extractor = FeatureSequenceOutput(
            vocab_path=vocab_path,
            matrix_dir=matrix_dir,  # Directory input for layered mode
            config_path=cam4_op.config_path,
            use_layered=True
        )
    else:
        # Single-layer mode (specific file path for matrix)
        extractor = FeatureSequenceOutput(
            vocab_path=vocab_path,
            cooccur_matrix_path=matrix_file,  # Specific compressed file path
            config_path=cam4_op.config_path,
            use_layered=False
        )

    print("  Extracting CAM features (full dataset)...")
    # Extract features with progress bar
    X = np.array([extractor.transform(t, return_sequence=False) for t in tqdm(all_texts, desc="  Features")])

    # Label encoding (convert string labels to integers)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(all_labels)

    # Feature standardization (zero mean, unit variance)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===== Cross-validation for optimal CCP pruning parameter =====
    print("\n  Selecting optimal pruning parameters via 5-fold cross-validation...")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    # Grow initial unpruned tree (larger than final model)
    full_tree = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='entropy',
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    full_tree.fit(X_scaled, y)
    print(f"    Initial tree: {full_tree.get_n_leaves()} leaves, depth {full_tree.get_depth()}")

    # Get CCP pruning path
    path = full_tree.cost_complexity_pruning_path(X_scaled, y)
    ccp_alphas = path.ccp_alphas

    # Select representative alpha values for cross-validation
    unique_alphas = np.unique(ccp_alphas[ccp_alphas >= 0])
    if len(unique_alphas) > 10:
        # Sample 10 representative values if too many
        indices = np.linspace(0, len(unique_alphas) - 1, 10, dtype=int)
        test_alphas = unique_alphas[indices]
    else:
        test_alphas = unique_alphas

    # Find alpha with best cross-validation score
    best_alpha = 0.0
    best_cv_score = 0.0

    print(f"    Testing {len(test_alphas)} pruning parameters...")
    for alpha in tqdm(test_alphas, desc="    CV Iterations"):
        if alpha < 0:
            continue

        # Train tree with current alpha
        tree = DecisionTreeClassifier(
            random_state=42,
            ccp_alpha=float(alpha),
            class_weight='balanced'
        )

        # 5-fold cross-validation
        scores = cross_val_score(tree, X_scaled, y, cv=5, scoring='accuracy', n_jobs=2)
        cv_mean = scores.mean()

        # Update best alpha if current is better
        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_alpha = float(alpha)

    print(f"    Optimal CCP alpha: {best_alpha:.6f}")
    print(f"    CV accuracy (In-Distribution): {best_cv_score:.2%}")
    print(f"    ⚠️  Note: CV accuracy is in-distribution - real OOD performance may be lower")

    # ===== Train final model on full dataset =====
    print("\n  Training final model on full dataset...")
    final_tree = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=best_alpha,
        class_weight='balanced'
    )
    final_tree.fit(X_scaled, y)

    # Create feature names (different for hierarchical mode)
    if use_layered:
        # Hierarchical feature naming (Word15 + Phrase15 + Sentence5)
        feature_names = [f"Word_{i}" for i in range(15)] + \
                        [f"Phrase_{i}" for i in range(15)] + \
                        [f"Sent_{i}" for i in range(5)]
    else:
        # Single-layer feature naming
        feature_names = [f"Feat_{i}" for i in range(expected_dim)]

    # Wrap tree for whitebox explainability
    clf = WhiteBoxTree(final_tree, feature_names, le.classes_)

    # Evaluate on full training set (in-distribution accuracy)
    train_acc = clf.score(X_scaled, y)
    n_leaves = final_tree.get_n_leaves()

    # Training summary
    print(f"\n{'=' * 60}")
    print(f"  Training Complete (Full Dataset)")
    print(f"  Training set accuracy (In-Distribution): {train_acc:.2%}")
    print(f"  ⚠️  Warning: This is in-distribution fit - not real generalization performance!")
    print(f"  Number of decision rules: {n_leaves}")
    print(f"  Tree depth: {final_tree.get_depth()}")
    print(f"  CCP pruning alpha: {best_alpha:.6f}")

    # Display top 5 decision rules (whitebox explainability)
    print(f"\n  Whitebox Explanation: Top 5 Core Decision Rules")
    sorted_rules = sorted(clf.rules, key=lambda x: x['samples'], reverse=True)
    for i, rule in enumerate(sorted_rules[:5], 1):
        # Truncate long condition lists for readability
        cond = " AND ".join(rule['conditions'][:2])
        if len(rule['conditions']) > 2:
            cond += f" ...({len(rule['conditions']) - 2} more conditions)"
        print(f"    Rule {i} [{rule['samples']} samples]: IF {cond}")
        print(f"           THEN {rule['prediction']} (confidence {rule['probability']:.1%})")

    # ===== Save model artifacts with compression =====
    print(f"\n  Saving model (joblib compression level: {model_compress_level})...")

    # Save decision tree (main model - configurable compression)
    tree_path = os.path.join(output_dir, 'whitebox_tree.pkl')
    joblib.dump(clf, tree_path, compress=('lzma', model_compress_level))

    # Save scaler (small file - fast compression)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path, compress=('lzma', 1))

    # Save label encoder (small file - fast compression)
    le_path = os.path.join(output_dir, 'label_encoder.pkl')
    joblib.dump(le, le_path, compress=('lzma', 1))

    # Create comprehensive metadata (JSON-serializable)
    meta = {
        'feature_type': f"layered_CAM-{expected_dim}" if use_layered else f"single_CAM-{expected_dim}",
        'feature_dim': int(expected_dim),
        'use_layered': bool(use_layered),
        'train_accuracy_in_distribution': float(train_acc),
        'cv_accuracy_in_distribution': float(best_cv_score),
        'ccp_alpha': float(best_alpha),
        'n_rules': int(n_leaves),
        'tree_depth': int(final_tree.get_depth()),
        'n_samples': int(len(X)),
        'warning': 'Accuracies are in-distribution estimates. Real OOD performance may be significantly lower.',
        'model_type': 'WhiteBox Decision Tree (CCP Pruned)',
        'compression': {
            'algorithm': 'lzma',
            'level': int(model_compress_level),
            'scaler_level': 1,
            'encoder_level': 1
        }
    }

    # Ensure all values are JSON-serializable
    meta = convert_to_serializable(meta)

    # Save metadata
    with open(os.path.join(output_dir, 'classifier_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Save decision rules (for whitebox explainability)
    rules_export = {
        'metadata': {
            'train_acc': float(train_acc),
            'cv_acc': float(best_cv_score),
            'n_rules': int(n_leaves),
            'tree_depth': int(final_tree.get_depth()),
            'ccp_alpha': float(best_alpha)
        },
        'top_rules': convert_to_serializable(sorted_rules[:30]),  # Top 30 rules
        'feature_names': feature_names,
        'classes': [str(c) for c in le.classes_]
    }

    with open(os.path.join(output_dir, 'decision_rules.json'), 'w', encoding='utf-8') as f:
        json.dump(rules_export, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n  Model and rules saved successfully")
    print(f"  Model size: {Path(tree_path).stat().st_size / 1024:.1f} KB")
    print(f"  Interpretability: View decision_rules.json for complete rule set")

    return clf, scaler, le


def main():
    """
    Main CLI entry point for CAM whitebox training.

    Command-line Arguments:
        --data: Training data directory (default: Data/train)
        --config: Configuration file path (default: Config/System_Config.json)
        --layered: Enable hierarchical CAM-35 mode (flag)
        --output-base: Base directory for output (default: Output/train)
        --model-compress-level: Model compression level (0-9, default=3)

    Workflow:
        1. Parse command-line arguments
        2. Initialize CAM operator with configuration
        3. Build CAM pipeline (stats → vocab → matrices)
        4. Train whitebox decision tree on full dataset
        5. Save model artifacts with compression
        6. Generate explainability reports
    """
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description='CAM WhiteBox Training (Full Data + Compression Support)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core arguments
    parser.add_argument('--data',
                        default=os.path.join(PROJECT_ROOT, 'Data/train'),
                        help='Training data directory (default: Data/train)')
    parser.add_argument('--config',
                        default=os.path.join(PROJECT_ROOT, 'Config/System_Config.json'),
                        help='Configuration file path (default: Config/System_Config.json)')
    parser.add_argument('--layered',
                        action='store_true',
                        help='Enable hierarchical CAM-35 mode (Word+Phrase+Sentence layers)')
    parser.add_argument('--output-base',
                        default=os.path.join(PROJECT_ROOT, 'Output/train'),
                        help='Base directory for output (default: Output/train)')
    parser.add_argument('--model-compress-level',
                        type=int,
                        default=3,
                        help='Model compression level (0=none to 9=maximum, default=3)')

    # Parse arguments
    args = parser.parse_args()

    # Initial setup
    print("=" * 60)
    print("CAM-15/35 Whitebox Decision Tree Training (Full Dataset Version)")
    print("⚠️  No train/validation split | Cross-validation parameter selection")
    print("⚠️  Optimized compressed matrix format support (.xz, .joblib, .npz)")
    print("=" * 60)

    # Create unique run directory
    run_dir = Path(args.output_base) / f"run{len(list(Path(args.output_base).glob('run*'))) + 1}"
    run_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {run_dir}")

    try:
        # Load and update configuration
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['use_layered_cam'] = args.layered

        # Save temporary training configuration
        temp_config = run_dir / "train_config.json"
        with open(temp_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Initialize CAM operator
        cam4_op = CAM4Operator(str(temp_config))

        # Build complete CAM pipeline (stats → vocab → matrices)
        results = cam4_op.build_pipeline(args.data, str(run_dir))

        # Train whitebox decision tree
        clf, scaler, le = train_whitebox_full(
            cam4_op=cam4_op,
            vocab_path=results['vocab'],
            matrix_dir=str(run_dir),
            data_dir=args.data,
            output_dir=str(run_dir),
            use_layered=args.layered,
            model_compress_level=args.model_compress_level
        )

        # Final success message
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("✓ Parameters selected via cross-validation (no validation set leakage)")
        print("✓ Optimized LZMA compression for matrices and model files")
        print("✓ IMPORTANT: Validate real OOD performance on independent test set!")
        print("=" * 60)

    except Exception as e:
        # Error handling
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    """Main entry point - exit with return code"""
    sys.exit(main())