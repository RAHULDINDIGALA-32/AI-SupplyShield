import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Compatibility shim: CorrelationPruner was defined in the training notebook
class CorrelationPruner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.keep_columns_: pd.Index | None = None

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X).copy()
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_columns_ = X_df.columns.difference(to_drop)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        if self.keep_columns_ is None:
            return X_df
        cols = [c for c in X_df.columns if c in set(self.keep_columns_)]
        # If no columns match by name, try positional matching (by index)
        # This handles cases where column names don't match but structure does
        if len(cols) == 0 and len(self.keep_columns_) > 0:
            # Try to use columns by position if we have the right number
            if X_df.shape[1] >= len(self.keep_columns_):
                # Use first N columns where N is the number of kept columns
                return X_df.iloc[:, :len(self.keep_columns_)]
            else:
                # Not enough columns - return what we have
                return X_df
        return X_df[cols] if len(cols) > 0 else X_df


# -------------------------
# Model Loading Functions
# -------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()

DEFAULT_SEARCH_DIRS = [
    SCRIPT_DIR,
    Path("./ai_supplyshield_qut_consolidated"),
    Path("./artifacts"),
    Path("/kaggle/working/ai_supplyshield_qut_consolidated"),
]


def find_manifest() -> Path | None:
    """Find the hybrid manifest file."""
    for base in DEFAULT_SEARCH_DIRS:
        cand_combined = base / "hybrid_manifest_combined.json"
        if cand_combined.exists():
            return cand_combined
        cand_pertrace = base / "hybrid_manifest.json"
        if cand_pertrace.exists():
            return cand_pertrace
    return None


def _patch_tree_nodes_array(nodes):
    """Patch a tree nodes array to add missing_go_to_left field if missing."""
    if nodes is None or len(nodes) == 0:
        return nodes

    if hasattr(nodes, 'dtype') and hasattr(nodes.dtype, 'names'):
        if 'missing_go_to_left' in nodes.dtype.names:
            return nodes

        new_dtype = [
            ('left_child', '<i8'),
            ('right_child', '<i8'),
            ('feature', '<i8'),
            ('threshold', '<f8'),
            ('impurity', '<f8'),
            ('n_node_samples', '<i8'),
            ('weighted_n_node_samples', '<f8'),
            ('missing_go_to_left', 'u1'),
        ]

        new_nodes = np.zeros(len(nodes), dtype=new_dtype)
        for field in nodes.dtype.names:
            new_nodes[field] = nodes[field]
        new_nodes['missing_go_to_left'] = 0
        
        return new_nodes

    return nodes


def load_model_with_compatibility(model_path: Path):
    """Load a sklearn model with compatibility handling for version mismatches."""
    # Try normal loading first
    try:
        return joblib.load(model_path, mmap_mode=None)
    except (ValueError, TypeError, AttributeError) as e:
        error_str = str(e)
        
        # Check if it's the missing_go_to_left error (various error message formats)
        is_dtype_error = (
            "incompatible dtype" in error_str or 
            "missing_go_to_left" in error_str or
            "node array from the pickle" in error_str or
            ("expected:" in error_str.lower() and "got :" in error_str.lower()) or
            "cannot set 'setstate'" in error_str
        )
        
        if is_dtype_error:
            # The simplest solution: recommend downgrading sklearn to match model version
            # Models were saved with sklearn 1.2.2, current is 1.3.2 or higher
            raise ValueError(
                f"Model loading failed due to sklearn version incompatibility.\n"
                f"Models were saved with sklearn 1.2.2, but you're using a different version.\n\n"
                f"**Recommended Solution:**\n"
                f"Downgrade scikit-learn to match the model version:\n"
                f"  pip install scikit-learn==1.2.2\n\n"
                f"Or try a compatible version:\n"
                f"  pip install scikit-learn==1.3.2\n\n"
                f"**Alternative:** Re-save your models with the current sklearn version.\n\n"
                f"Original error: {error_str}"
            )
        else:
            # Not the error we're trying to fix, re-raise
            raise


def load_artifacts(manifest_path: Path):
    """Load all model artifacts from the manifest."""
    base = manifest_path.parent
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    best_trace = manifest["best_trace"]
    art = manifest["artifacts"]

    try:
        pipeline = load_model_with_compatibility(base / art["pipeline"])
        rf = load_model_with_compatibility(base / art["rf"])
        isof = load_model_with_compatibility(base / art["isof"])
        fusion = load_model_with_compatibility(base / art["fusion"])
    except Exception as e:
        error_msg = str(e)
        if "NumpyUnpickler" in error_msg or "ensure_native_byte_order" in error_msg:
            raise ValueError(
                f"Model loading failed due to joblib version incompatibility.\n"
                f"Error: {error_msg}\n\n"
                f"Possible solutions:\n"
                f"1. Update joblib: pip install --upgrade joblib\n"
                f"2. Update scikit-learn: pip install --upgrade scikit-learn\n"
                f"3. Or try: pip install joblib==1.3.2 scikit-learn==1.3.2"
            ) from e
        else:
            raise
    
    # Load test stream if available
    test_stream = None
    if "test_stream" in art and (base / art["test_stream"]).exists():
        test_stream = pd.read_csv(base / art["test_stream"])
    
    return best_trace, pipeline, rf, isof, fusion, test_stream


# -------------------------
# Prediction Functions
# -------------------------
def predict_hybrid(X_processed: pd.DataFrame, pipeline, rf, isof, fusion, 
                   reference_scores: np.ndarray | None = None):
    """
    Make predictions using the hybrid model (vectorized for performance).
    
    Args:
        X_processed: Preprocessed features (after pipeline transformation)
        pipeline: Preprocessing pipeline
        rf: RandomForest model
        isof: IsolationForest model
        fusion: LogisticRegression fusion model
        reference_scores: Optional array of previous anomaly scores for ranking
    
    Returns:
        Dictionary with predictions and intermediate scores
    """
    # Vectorized predictions for better performance
    # Supervised prediction (RandomForest) - batch processing
    sup_probs = rf.predict_proba(X_processed)[:, 1]
    
    # Unsupervised prediction (IsolationForest) - batch processing
    # Negative because lower scores mean more anomalous
    uns_scores = -isof.score_samples(X_processed)
    
    # Rank anomaly scores against reference distribution (vectorized)
    if reference_scores is None or len(reference_scores) == 0:
        # Use current batch for ranking
        ranks = np.ones(len(uns_scores))  # Default to worst-case rank
    else:
        # Vectorized ranking: append all scores, rank, then extract ranks for new scores
        all_scores = np.append(reference_scores, uns_scores)
        all_ranks = pd.Series(all_scores).rank(pct=True).values
        # Extract ranks for the new scores (last len(uns_scores) elements)
        ranks = all_ranks[-len(uns_scores):]
    
    # Fusion prediction - batch processing
    fusion_input = np.column_stack([sup_probs, ranks])
    fused_probs = fusion.predict_proba(fusion_input)[:, 1]
    
    # Final predictions
    y_preds = (fused_probs >= 0.5).astype(int)
    
    # Convert to lists for compatibility
    results = {
        'supervised_prob': sup_probs.tolist(),
        'anomaly_score': uns_scores.tolist(),
        'anomaly_rank': ranks.tolist(),
        'fused_prob': fused_probs.tolist(),
        'prediction': y_preds.tolist()
    }
    
    return results


# -------------------------
# Visualization Functions
# -------------------------
def plot_confusion_matrix(y_true, y_pred, ax=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax.figure if ax is None else None


def plot_roc_curve(y_true, y_score, ax=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax.figure if ax is None else None


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="AI Supply Shield — Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Supply Shield - Trojanized Package Detection")
st.markdown("**Hybrid Model:** RandomForest + IsolationForest fused via Logistic Regression")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Find and load manifest
    default_manifest = find_manifest()
    user_dir = st.text_input(
        "Artifacts Directory",
        value=str(default_manifest.parent) if default_manifest else str(SCRIPT_DIR),
        help="Directory containing the model artifacts"
    )
    
    manifest_path = None
    if user_dir:
        manifest_candidate = Path(user_dir) / "hybrid_manifest_combined.json"
        if manifest_candidate.exists():
            manifest_path = manifest_candidate
        else:
            manifest_candidate = Path(user_dir) / "hybrid_manifest.json"
            if manifest_candidate.exists():
                manifest_path = manifest_candidate
    
    if not manifest_path or not manifest_path.exists():
        st.error("❌ Manifest file not found. Please ensure `hybrid_manifest_combined.json` exists in the specified directory.")
        st.stop()

    # Load models
    try:
        best_trace, pipeline, rf, isof, fusion, test_stream = load_artifacts(manifest_path)
        st.success(f"✅ Loaded models for: **{best_trace}**")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

    st.divider()
    st.header("📊 Input Options")
    input_mode = st.radio(
        "Select input method:",
        ["Upload CSV", "Use Test Data", "Manual Input"],
        help="Choose how to provide data for prediction"
    )

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Analysis", "ℹ️ About"])

with tab1:
    # Data input section
    st.header("Data Input")
    
    X_raw = None
    y_true = None
    
    if input_mode == "Upload CSV":
        st.info("📝 **Note:** For best results, upload preprocessed data with columns: `f_0`, `f_1`, `f_2`, ... `f_48` (49 features, values 0-1). Raw trace features may not work due to feature name mismatches.")
        uploaded_file = st.file_uploader(
            "Upload CSV file with features",
            type=['csv'],
            help="Upload a CSV file with preprocessed features (f_0, f_1, ... f_48) or raw features (may require exact column names)"
        )
        
        if uploaded_file is not None:
            try:
                X_raw = pd.read_csv(uploaded_file)
                # Check if label column exists
                label_cols = [c for c in X_raw.columns if c.lower() in ['label', 'target', 'class', 'y']]
                if label_cols:
                    y_true = X_raw[label_cols[0]]
                    X_raw = X_raw.drop(columns=label_cols)
                    st.info(f"Found label column: {label_cols[0]}")
                
                # Check if data is already preprocessed (has f_0, f_1, etc. columns)
                numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
                is_preprocessed_format = (
                    len(numeric_cols) > 0 and
                    all(str(col).startswith('f_') or str(col).startswith('feature_') for col in numeric_cols[:5])
                )
                
                if is_preprocessed_format:
                    st.session_state['is_preprocessed'] = True
                    st.info("ℹ️ Detected preprocessed data format - will skip pipeline transformation")
                
                st.success(f"✅ Loaded {len(X_raw)} samples with {len(X_raw.columns)} features")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    elif input_mode == "Use Test Data":
        if test_stream is not None:
            if st.button("Load Test Data"):
                X_raw = test_stream.drop(columns=['label'], errors='ignore')
                y_true = test_stream['label'] if 'label' in test_stream.columns else None
                # Mark that this is preprocessed data (test_stream is already post-pipeline)
                st.session_state['is_preprocessed'] = True
                st.success(f"✅ Loaded {len(X_raw)} test samples (already preprocessed)")
        else:
            st.warning("⚠️ Test data not available in manifest")
    
    elif input_mode == "Manual Input":
        st.info("Manual input mode - coming soon. Please use CSV upload or test data for now.")
    
    # Prediction section
    if X_raw is not None and len(X_raw) > 0:
        st.divider()
        st.header("🔮 Predictions")
        
        # Preprocess data
        try:
            # Check if data is already preprocessed (test_stream case)
            is_preprocessed = st.session_state.get('is_preprocessed', False)
            
            if is_preprocessed:
                # Test data is already preprocessed, use directly
                with st.spinner("Using preprocessed test data..."):
                    X_processed = X_raw.select_dtypes(include=[np.number])
                    # Keep original column names (f_0, f_1, etc.) - models expect these exact names
                    if X_processed.shape[1] == 0:
                        raise ValueError("No numeric columns found in test data")
                    # Test data should already have f_0, f_1, etc. - keep them as-is
                    # Only rename if they don't match the expected pattern
                    current_cols = list(X_processed.columns)
                    if not all(str(col).startswith('f_') for col in current_cols):
                        # Rename to match model expectations
                        X_processed.columns = [f"f_{i}" for i in range(X_processed.shape[1])]
                st.success(f"✅ Using {X_processed.shape[1]} preprocessed features")
            else:
                # Raw data needs preprocessing
                with st.spinner("Preprocessing data..."):
                    # Ensure numeric columns only
                    X_numeric = X_raw.select_dtypes(include=[np.number])
                    
                    if X_numeric.shape[1] == 0:
                        raise ValueError("No numeric columns found. Please ensure your CSV contains numeric feature columns.")
                    
                    # Handle missing values
                    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
                    X_numeric = X_numeric.fillna(X_numeric.median())
                    
                    # Apply pipeline
                    # The pipeline expects specific feature names, but we have raw trace features
                    # We need to bypass sklearn's feature name validation
                    try:
                        # Convert to numpy array to bypass feature name validation
                        X_array_input = X_numeric.values
                        
                        # Temporarily patch sklearn's feature name validation to skip checking
                        from sklearn.base import BaseEstimator
                        original_check_feature_names = None
                        
                        # Transform through pipeline steps manually to bypass feature name validation
                        if hasattr(pipeline, 'named_steps'):
                            steps = pipeline.named_steps
                            X_temp = X_array_input
                            
                            # Step 1: CorrelationPruner
                            if 'prune' in steps:
                                pruner = steps['prune']
                                # Create DataFrame with generic column names for pruner
                                X_df_temp = pd.DataFrame(X_temp, columns=[f"col_{i}" for i in range(X_temp.shape[1])])
                                X_pruned = pruner.transform(X_df_temp)
                                X_temp = X_pruned.values if isinstance(X_pruned, pd.DataFrame) else X_pruned
                            
                            # Step 2: MinMaxScaler - use array directly to bypass feature name check
                            if 'scale' in steps:
                                scaler = steps['scale']
                                # Use _validate_data=False or work with array directly
                                # The scaler's transform should work with arrays
                                try:
                                    # Try normal transform
                                    X_temp = scaler.transform(X_temp)
                                except ValueError as ve:
                                    if "feature names" in str(ve).lower():
                                        # Bypass by using internal methods
                                        # Access the scaler's internal state and transform directly
                                        X_temp = (X_temp - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_)
                                        # Handle any division by zero
                                        X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=1.0, neginf=0.0)
                                    else:
                                        raise
                            
                            X_array = X_temp
                        else:
                            # Fallback: try direct transform with array
                            X_array = pipeline.transform(X_array_input)
                            
                        # Check if pipeline returned empty array
                        if X_array.shape[1] == 0 or len(X_array.shape) < 2:
                            raise ValueError("Pipeline returned empty or invalid output")
                            
                    except Exception as pipe_error:
                        error_msg = str(pipe_error)
                        # If it's a feature name error, provide helpful message
                        if "feature names" in error_msg.lower() or "unseen at fit time" in error_msg.lower():
                            raise ValueError(
                                f"**Feature Name Mismatch Error**\n\n"
                                f"The pipeline was trained on different feature names than your CSV.\n"
                                f"Your CSV has: {', '.join(X_numeric.columns[:5].tolist())}...\n\n"
                                f"**Recommended Solutions:**\n\n"
                                f"1. **Use Preprocessed Data**: Select 'Use Test Data' option\n"
                                f"   (This uses data that's already in the correct format)\n\n"
                                f"2. **Upload Preprocessed CSV**: Your CSV should have columns:\n"
                                f"   `f_0`, `f_1`, `f_2`, ... `f_48` (49 features, values 0-1)\n\n"
                                f"3. **Match Training Features**: Ensure your CSV has the exact\n"
                                f"   same feature names as used during training\n\n"
                                f"**Note:** The pipeline expects preprocessed features from the\n"
                                f"combined trace dataset, not raw individual trace features.\n\n"
                                f"Original error: {error_msg[:200]}..."
                            ) from pipe_error
                        else:
                            raise ValueError(
                                f"Pipeline transformation failed: {error_msg}\n\n"
                                "This may happen if:\n"
                                "1. Column structure doesn't match training data\n"
                                "2. All features were filtered out\n"
                                "3. Data format is incorrect\n\n"
                                "Please ensure your CSV has the same number and type of features as the training data."
                            ) from pipe_error
                    
                    # Use f_0, f_1, etc. to match what models expect
                    X_processed = pd.DataFrame(
                        X_array,
                        columns=[f"f_{i}" for i in range(X_array.shape[1])]
                    )
                
                st.success(f"✅ Preprocessed to {X_processed.shape[1]} features")
            
            # Reset the flag for next time
            st.session_state['is_preprocessed'] = False
            
            # Make predictions
            with st.spinner("Making predictions..."):
                # Use test stream scores as reference if available
                reference_scores = None
                if test_stream is not None and 'label' in test_stream.columns:
                    # Test stream is already preprocessed, use directly
                    test_features = test_stream.drop(columns=['label']).select_dtypes(include=[np.number])
                    if test_features.shape[1] > 0:
                        # Ensure column names match what models expect (f_0, f_1, etc.)
                        current_cols = list(test_features.columns)
                        if not all(str(col).startswith('f_') for col in current_cols):
                            test_features.columns = [f"f_{i}" for i in range(test_features.shape[1])]
                        reference_scores = -isof.score_samples(test_features)
                
                results = predict_hybrid(X_processed, pipeline, rf, isof, fusion, reference_scores)
                
                # Store in session state for use in other tabs
                st.session_state['prediction_results'] = results
                st.session_state['y_true'] = y_true.values if y_true is not None else None
                st.session_state['X_processed'] = X_processed
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Supervised_Prob': results['supervised_prob'],
                    'Anomaly_Score': results['anomaly_score'],
                    'Anomaly_Rank': results['anomaly_rank'],
                    'Fused_Prob': results['fused_prob'],
                    'Prediction': ['Malicious' if p == 1 else 'Benign' for p in results['prediction']]
                })
                
                if y_true is not None:
                    results_df['True_Label'] = ['Malicious' if y == 1 else 'Benign' for y in y_true]
                    results_df['Correct'] = (results['prediction'] == y_true.values)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                malicious_count = sum(results['prediction'])
                benign_count = len(results['prediction']) - malicious_count
                st.metric("Malicious", malicious_count)
            
            with col2:
                st.metric("Benign", benign_count)
            
            with col3:
                if y_true is not None:
                    accuracy = accuracy_score(y_true, results['prediction'])
                    st.metric("Accuracy", f"{accuracy:.3f}")
                else:
                    st.metric("Fused Prob (Avg)", f"{np.mean(results['fused_prob']):.3f}")
            
            with col4:
                if y_true is not None:
                    f1 = f1_score(y_true, results['prediction'])
                    st.metric("F1 Score", f"{f1:.3f}")
                else:
                    st.metric("Malicious Rate", f"{malicious_count/len(results['prediction']):.3f}")
            
            # Detailed metrics if labels available
            if y_true is not None:
                st.subheader("📊 Performance Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    precision = precision_score(y_true, results['prediction'], zero_division=0)
                    recall = recall_score(y_true, results['prediction'], zero_division=0)
                    f1 = f1_score(y_true, results['prediction'])
                    roc_auc = roc_auc_score(y_true, results['fused_prob'])
                    
                    st.write(f"**Precision:** {precision:.3f}")
                    st.write(f"**Recall:** {recall:.3f}")
                    st.write(f"**F1 Score:** {f1:.3f}")
                    st.write(f"**ROC-AUC:** {roc_auc:.3f}")
                
                with metrics_col2:
                    st.write(f"**Total Samples:** {len(y_true)}")
                    st.write(f"**True Positives:** {sum((y_true == 1) & (np.array(results['prediction']) == 1))}")
                    st.write(f"**True Negatives:** {sum((y_true == 0) & (np.array(results['prediction']) == 0))}")
                    st.write(f"**False Positives:** {sum((y_true == 0) & (np.array(results['prediction']) == 1))}")
                    st.write(f"**False Negatives:** {sum((y_true == 1) & (np.array(results['prediction']) == 0))}")
            
            # Display results table
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)

with tab2:
    st.header("📈 Analysis & Visualizations")
    
    if 'prediction_results' in st.session_state:
        results = st.session_state['prediction_results']
        y_true = st.session_state.get('y_true', None)
        # Check if we have true labels for evaluation
        if y_true is not None:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_confusion_matrix(y_true, results['prediction'], ax=ax)
            st.pyplot(fig)
            
            # ROC Curve
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_roc_curve(y_true, results['fused_prob'], ax=ax)
            st.pyplot(fig)
        
        # Probability distributions
        st.subheader("Probability Distributions")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(results['supervised_prob'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Supervised Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('RandomForest Probability Distribution')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(results['fused_prob'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Fused Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Hybrid Fusion Probability Distribution')
        axes[1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Anomaly score distribution
        st.subheader("Anomaly Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(results['anomaly_score'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Anomaly Score (higher = more anomalous)')
        ax.set_ylabel('Frequency')
        ax.set_title('IsolationForest Anomaly Score Distribution')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Time series of predictions (if sequential)
        if len(results['fused_prob']) > 1:
            st.subheader("Prediction Timeline")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(results['fused_prob'], label='Fused Probability', alpha=0.7)
            ax.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
            if y_true is not None:
                ax.scatter(range(len(y_true)), y_true, 
                          c=['red' if y == 1 else 'green' for y in y_true],
                          alpha=0.3, s=20, label='True Labels')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probability Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    else:
        st.info("👆 Please load data and make predictions in the 'Prediction' tab first.")

with tab3:
    st.header("ℹ️ About AI Supply Shield")
    
    st.markdown("""
    ### Model Architecture
    
    This application uses a **hybrid ensemble approach** combining supervised and unsupervised learning:
    
    1. **RandomForest (Supervised)**: Trained on labeled data to classify packages as benign or malicious
    2. **IsolationForest (Unsupervised)**: Detects anomalies without requiring labels
    3. **Logistic Regression (Fusion)**: Combines predictions from both models for final decision
    
    ### Preprocessing Pipeline
    
    The data undergoes the following preprocessing steps:
    - **Correlation Pruning**: Removes highly correlated features (|r| > 0.9)
    - **MinMax Scaling**: Normalizes features to [0, 1] range
    
    ### How It Works
    
    1. Input data is preprocessed using the fitted pipeline
    2. RandomForest provides a supervised probability score
    3. IsolationForest provides an anomaly score (higher = more anomalous)
    4. The anomaly score is ranked against reference data
    5. Both scores are fed into a Logistic Regression fusion model
    6. Final prediction is made based on the fused probability (threshold = 0.5)
    
    ### Model Files
    
    The following artifacts are required:
    - `pipeline_combined.pkl`: Preprocessing pipeline
    - `rf_hybrid_Combined.joblib`: RandomForest model
    - `isof_hybrid_Combined.joblib`: IsolationForest model
    - `fusion_hybrid_Combined.joblib`: LogisticRegression fusion model
    - `hybrid_manifest_combined.json`: Manifest file referencing all artifacts
    
    ### Usage
    
    1. **Upload CSV**: Upload a CSV file with feature columns matching the training data
    2. **Use Test Data**: Load the provided test dataset for evaluation
    3. **View Results**: Check predictions, metrics, and visualizations
    
    ### Output
    
    - **Prediction**: Binary classification (Benign/Malicious)
    - **Fused Probability**: Final probability score from the hybrid model
    - **Supervised Probability**: Score from RandomForest
    - **Anomaly Score**: Score from IsolationForest
    - **Anomaly Rank**: Percentile rank of the anomaly score
    """)
    
    st.divider()
    st.markdown("**Built for detecting trojanized Python packages in software supply chains**")

