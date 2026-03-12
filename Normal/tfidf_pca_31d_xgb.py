#!/usr/bin/env python3
"""
TF-IDF+PCA Dimensionality Reduction to 31/15D + XGBoost Comparison Experiment
Author: ChaoJiht666
Repository: GitHub (CAM-15/35)

Core Features:
- Single CSV file/directory loading support with encoding validation
- Consistent text cleaning with CAM operator
- TF-IDF feature extraction + PCA dimensionality reduction (31/15D)
- XGBoost classifier with 5-fold cross-validation
- Comprehensive training/evaluation pipeline
- Detailed test results with accuracy statistics
- Model serialization for reproducibility

Key Fixes:
- Added encoding tolerance (UTF-8/GBK) for Chinese text
- Column validation for 'text' and 'title' fields
- Single CSV file loading support (in addition to directory)
- Min_df adjustment for small dataset compatibility
- Robust error handling for data loading and processing
"""

import datetime
import sys
import os
import traceback
import string
import json
from pathlib import Path
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import argparse

# Scikit-learn imports for feature processing and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

# XGBoost with fallback
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not found, using GradientBoostingClassifier as fallback")

# Project configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def clean_text(text):
    """
    Text cleaning function - identical to CAM operator implementation.

    Removes all punctuation (Chinese + English) while preserving other characters.
    Critical for maintaining consistency with CAM pipeline.

    Args:
        text: Input text string to clean

    Returns:
        str: Cleaned text with punctuation removed
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


def setup_run_directory(base_dir: str) -> str:
    """
    Create sequentially numbered run directory (consistent with CAM operator).

    Args:
        base_dir: Base directory for run storage

    Returns:
        str: Path to newly created run directory (e.g., "run1", "run2")
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Count existing run directories
    existing_runs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('run')]
    run_id = len(existing_runs) + 1
    run_dir = base / f"run{run_id}"
    run_dir.mkdir(exist_ok=True)

    return str(run_dir)


def load_data(data_path):
    """
    Robust data loading with validation and error handling.

    Features:
        - Support for single CSV file or directory of CSV files
        - Encoding tolerance (UTF-8/GBK) for Chinese text
        - Column validation for 'text' and 'title' fields
        - Empty value filtering
        - Consistent text cleaning with CAM operator
        - Basic dataset statistics

    Args:
        data_path: Path to CSV file or directory containing CSV files

    Returns:
        tuple: (list of cleaned texts, list of corresponding labels)
    """
    all_texts, all_titles = [], []
    data_path = Path(data_path)

    # Determine files to process
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        print(f"📂 Found {len(csv_files)} CSV files in directory")
    elif data_path.suffix == '.csv':
        csv_files = [data_path]
        print(f"📄 Processing single CSV file: {data_path}")
    else:
        print(f"❌ Error: {data_path} is not a CSV file or directory")
        return all_texts, all_titles

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Handle different encodings for Chinese text
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='gbk')
                print(f"  ⚠️  Using GBK encoding for {csv_file.name}")

            # Validate required columns
            if 'text' not in df.columns or 'title' not in df.columns:
                print(f"  ⚠️  Skipping {csv_file.name}: Missing 'text' or 'title' column")
                continue

            # Process each row
            valid_rows = 0
            for idx, row in df.iterrows():
                # Extract and validate text and label
                raw_text = str(row['text']) if not pd.isna(row['text']) else ""
                label = str(row['title']) if not pd.isna(row['title']) else ""

                # Skip empty values
                if not raw_text.strip() or not label.strip():
                    continue

                # Clean text (consistent with CAM operator)
                cleaned_text = clean_text(raw_text)
                if not cleaned_text:
                    continue

                # Add to dataset
                all_texts.append(cleaned_text)
                all_titles.append(label)
                valid_rows += 1

            print(f"  ✅ {csv_file.name}: Loaded {valid_rows} valid rows")

        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {str(e)[:100]}")
            continue

    # Dataset statistics
    print(f"\n📊 Dataset Summary:")
    print(f"   Total valid samples: {len(all_texts)}")

    if all_titles:
        label_counter = Counter(all_titles)
        print(f"   Number of classes: {len(label_counter)}")
        print(f"   Top 5 classes by count: {dict(label_counter.most_common(5))}")
    else:
        print("   ⚠️  No valid samples loaded")

    return all_texts, all_titles


def train_tfidf_pca_xgb(data_dir, output_dir, n_components=31):
    """
    Complete training pipeline: TF-IDF → PCA → XGBoost.

    Pipeline steps:
        1. Data loading and validation
        2. Label encoding
        3. TF-IDF feature extraction (character-level, 1-2 ngrams)
        4. PCA dimensionality reduction to specified dimensions
        5. Feature standardization
        6. XGBoost training with 5-fold cross-validation
        7. Model serialization
        8. Training metadata saving

    Args:
        data_dir: Path to training data (CSV file/directory)
        output_dir: Directory to save trained models and metadata
        n_components: PCA target dimensionality (default: 31 to match CAM)

    Returns:
        tuple: (trained classifier, training metadata dictionary)
    """
    print("\n" + "=" * 70)
    print(f"🚀 Starting TF-IDF+PCA({n_components}D)+XGBoost Training")
    print("=" * 70)

    # Step 1: Load and validate data
    texts, labels = load_data(data_dir)
    if len(texts) == 0:
        print("❌ Training aborted: No valid training data")
        return None, None

    # Step 2: Encode labels
    print("\n🔤 Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"   Number of classes: {n_classes}")

    # Step 3: TF-IDF feature extraction
    print("\n📝 Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,  # Adjusted for small dataset compatibility
        max_df=0.95,
        analyzer='char',  # Character-level for Chinese text
        stop_words=None,
        dtype=np.float32
    )

    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    print(f"   TF-IDF shape: {X_tfidf.shape} (samples × features)")

    # Step 4: PCA dimensionality reduction
    print(f"\n📉 Applying PCA dimensionality reduction to {n_components}D...")
    # Convert to dense matrix (PCA requires dense input)
    X_tfidf_dense = X_tfidf.toarray()

    # Initialize and fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_tfidf_dense)

    # PCA statistics
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   PCA output shape: {X_pca.shape}")
    print(f"   Explained variance ratio: {explained_variance:.4f} ({explained_variance * 100:.2f}%)")

    # Step 5: Feature standardization
    print("\n⚖️ Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # Step 6: XGBoost classifier training
    print("\n🧠 Training XGBoost classifier...")

    # Configure classifier
    if HAS_XGB:
        print("   Using XGBoost classifier (native implementation)")
        clf = XGBClassifier(
            objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
            num_class=n_classes if n_classes > 2 else None,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            verbosity=0
        )
    else:
        print("   Using GradientBoostingClassifier (scikit-learn fallback)")
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

    # 5-fold cross-validation (consistent with CAM operator)
    print("\n📊 Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded), 1):
        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Train on fold
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_pred)
        cv_scores.append(fold_accuracy)

        print(f"   Fold {fold}: {fold_accuracy:.4f} ({fold_accuracy * 100:.2f}%)")

    # Train final model on full dataset
    print("\n🏁 Training final model on full dataset...")
    clf.fit(X_scaled, y_encoded)
    train_accuracy = accuracy_score(y_encoded, clf.predict(X_scaled))

    # Step 7: Save models and components
    print("\n💾 Saving model components...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all components
    joblib.dump(clf, output_path / 'xgb_classifier.pkl')
    joblib.dump(tfidf_vectorizer, output_path / 'tfidf_vectorizer.pkl')
    joblib.dump(pca, output_path / 'pca_transformer.pkl')
    joblib.dump(scaler, output_path / 'scaler.pkl')
    joblib.dump(le, output_path / 'label_encoder.pkl')

    # Step 8: Save training metadata
    training_metadata = {
        "training_timestamp": datetime.datetime.now().isoformat(),
        "feature_pipeline": f"TF-IDF → PCA({n_components}D) → StandardScaler",
        "tfidf_parameters": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.95,
            "analyzer": "char"
        },
        "original_tfidf_dimension": X_tfidf.shape[1],
        "pca_components": n_components,
        "pca_explained_variance": float(explained_variance),
        "cross_validation": {
            "folds": 5,
            "mean_accuracy": float(np.mean(cv_scores)),
            "std_accuracy": float(np.std(cv_scores)),
            "fold_scores": [float(score) for score in cv_scores]
        },
        "training_accuracy": float(train_accuracy),
        "n_classes": n_classes,
        "classes": le.classes_.tolist(),
        "classifier_type": "XGBClassifier" if HAS_XGB else "GradientBoostingClassifier",
        "text_cleaning": "Punctuation removal (Chinese + English)"
    }

    with open(output_path / 'train_meta.json', 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"📊 Cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"🎯 Training set accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"📉 PCA explained variance: {explained_variance:.4f} ({explained_variance * 100:.2f}%)")
    print(f"💾 Models saved to: {output_dir}")
    print("=" * 70)

    return clf, training_metadata


def test_tfidf_pca_xgb(test_path: str, train_run: str, show_details: int = 30):
    """
    Comprehensive testing pipeline for TF-IDF+PCA+XGBoost model.

    Features:
        - Consistent with CAM operator testing logic
        - Detailed prediction output for first N samples
        - Progress tracking for large datasets
        - Comprehensive accuracy statistics
        - Result saving (CSV + JSON summary)

    Args:
        test_path: Path to test data (CSV file/directory)
        train_run: Name of trained run directory (e.g., "run1")
        show_details: Number of samples to show detailed output for (default: 30)
    """
    # Setup output directory
    test_output_base = Path(PROJECT_ROOT) / 'Output' / 'test_tfidf_pca_31d'
    current_test_dir = setup_run_directory(str(test_output_base))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Locate trained model directory
    train_dir = Path(PROJECT_ROOT) / 'Output' / 'train_tfidf_pca_31d' / train_run

    # Header
    print("\n" + "=" * 80)
    print("🧪 TF-IDF+PCA(31D)+XGBoost Testing")
    print("=" * 80)
    print(f"📁 Test output directory: {Path(current_test_dir).name}")
    print(f"🔧 Trained model: {train_run}")
    print(f"📄 Test data: {test_path}")
    print(f"📋 Detailed output for first {show_details} samples")
    print("-" * 80)

    # Load model components
    print("🔄 Loading model components...")
    try:
        # Core model components
        clf = joblib.load(train_dir / 'xgb_classifier.pkl')
        tfidf = joblib.load(train_dir / 'tfidf_vectorizer.pkl')
        pca = joblib.load(train_dir / 'pca_transformer.pkl')
        scaler = joblib.load(train_dir / 'scaler.pkl')
        le = joblib.load(train_dir / 'label_encoder.pkl')

        # Training metadata
        with open(train_dir / 'train_meta.json', 'r', encoding='utf-8') as f:
            train_meta = json.load(f)

        print("✅ Model components loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)[:100]}")
        traceback.print_exc()
        return

    # Load test data
    print("\n📥 Loading test data...")
    texts, true_labels = load_data(test_path)

    if len(texts) == 0:
        print("❌ Testing aborted: No valid test data")
        return

    # Test execution
    results = []
    total_samples = 0
    correct_predictions = 0

    print(f"\n🚀 Starting prediction on {len(texts)} test samples...")
    print("=" * 80)

    # Process each sample
    for idx, (raw_text, true_label) in enumerate(zip(texts, true_labels)):
        try:
            # Text cleaning (consistent with training)
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                continue

            # Feature extraction pipeline
            text_tfidf = tfidf.transform([cleaned_text])
            text_tfidf_dense = text_tfidf.toarray()
            text_pca = pca.transform(text_tfidf_dense)
            text_scaled = scaler.transform(text_pca)

            # Prediction
            pred_idx = clf.predict(text_scaled)[0]
            pred_label = le.inverse_transform([int(pred_idx)])[0]

            # Confidence score (if available)
            if HAS_XGB:
                confidence = float(np.max(clf.predict_proba(text_scaled)[0]))
            else:
                confidence = 0.0

            # Evaluation
            is_correct = (pred_label == true_label)
            total_samples += 1

            if is_correct:
                correct_predictions += 1

            # Store results
            result_item = {
                'index': total_samples,
                'raw_text': raw_text[:100],  # Truncate for readability
                'cleaned_text': cleaned_text[:80],
                'true_label': true_label,
                'predicted_label': pred_label,
                'is_correct': is_correct,
                'confidence': confidence,
                'status': '✅ Correct' if is_correct else '❌ Incorrect'
            }
            results.append(result_item)

            # Detailed output for first N samples
            if total_samples <= show_details:
                status_icon = "✅" if is_correct else "❌"
                print(f"\n[{total_samples}] {status_icon}")
                print(f"   Text: {raw_text[:60]}{'...' if len(raw_text) > 60 else ''}")
                print(f"   True: {true_label} | Predicted: {pred_label}")
                if HAS_XGB:
                    print(f"   Confidence: {confidence:.2%}")
                if not is_correct:
                    print(f"   ⚠️  Prediction error")

            # Progress update
            if total_samples % 100 == 0 and total_samples > show_details:
                current_accuracy = (correct_predictions / total_samples) * 100
                print(f"\n📊 Progress: {total_samples}/{len(texts)} | Accuracy: {current_accuracy:.1f}%")

        except Exception as e:
            print(f"\n❌ Error processing sample {idx + 1}: {str(e)[:100]}")
            continue

    # Calculate final statistics
    final_accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0

    # Test summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary (TF-IDF+PCA 31D + XGBoost)")
    print("=" * 80)
    print(f"Total valid samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_samples - correct_predictions}")
    print(f"Overall accuracy: {final_accuracy:.2f}%")
    print(f"PCA explained variance (training): {train_meta.get('pca_explained_variance', 0):.2%}")
    print("=" * 80)

    # Save results
    if results:
        # Detailed results CSV
        results_df = pd.DataFrame(results)
        csv_path = Path(current_test_dir) / 'detailed_test_results.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Summary JSON
        test_summary = {
            'test_run': Path(current_test_dir).name,
            'train_run': train_run,
            'test_timestamp': timestamp,
            'total_samples_processed': total_samples,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': total_samples - correct_predictions,
            'overall_accuracy': final_accuracy,
            'feature_pipeline': 'TF-IDF + PCA(31D) + StandardScaler',
            'classifier': 'XGBoost' if HAS_XGB else 'GradientBoostingClassifier',
            'training_pca_explained_variance': train_meta.get('pca_explained_variance', 0)
        }

        summary_path = Path(current_test_dir) / 'test_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to:")
        print(f"   Detailed results: {csv_path}")
        print(f"   Test summary: {summary_path}")


def main():
    """
    Main entry point with command-line interface.

    Supports two modes:
        1. train: Train TF-IDF+PCA+XGBoost model
        2. test: Test pre-trained model on new data
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='TF-IDF+PCA+XGBoost Text Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train new model with default 31D PCA
  python tfidf_pca_31d_xgb.py train --data /path/to/train_data --dim 31

  # Test trained model
  python tfidf_pca_31d_xgb.py test --test-file /path/to/test.csv --run run1 --show 50
        """
    )

    # Create subparsers for train/test modes
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode')

    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train TF-IDF+PCA+XGBoost model')
    train_parser.add_argument('--data', required=True, help='Path to training data (CSV file/directory)')
    train_parser.add_argument('--dim', type=int, default=31, help='PCA target dimensions (default: 31)')

    # Testing subparser
    test_parser = subparsers.add_parser('test', help='Test pre-trained TF-IDF+PCA+XGBoost model')
    test_parser.add_argument('--test-file', required=True, help='Path to test data (CSV file/directory)')
    test_parser.add_argument('--run', required=True, help='Name of trained run directory (e.g., run1)')
    test_parser.add_argument('--show', type=int, default=30, help='Number of samples to show details for')

    # Parse arguments
    args = parser.parse_args()

    # Execute selected mode
    if args.mode == 'train':
        # Setup training output directory
        train_output_dir = setup_run_directory(
            os.path.join(PROJECT_ROOT, 'Output', 'train_tfidf_pca_31d')
        )
        # Run training
        train_tfidf_pca_xgb(args.data, train_output_dir, args.dim)

    elif args.mode == 'test':
        # Run testing
        test_tfidf_pca_xgb(args.test_file, args.run, args.show)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)