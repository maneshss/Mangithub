import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CACHE_DIR = os.path.join(BASE_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", MPL_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class ModelConfig:
    name: str
    rnn_type: str  # "SimpleRNN", "LSTM", "GRU"
    embedding_dim: int
    rnn_units: int
    dropout: float
    recurrent_dropout: float
    learning_rate: float
    batch_size: int
    epochs: int


def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    keyword = df["keyword"].fillna("")
    location = df["location"].fillna("")
    text = df["text"].fillna("")
    return (keyword + " " + location + " " + text).str.strip()


def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    train_text = combine_text_fields(train_df)
    test_text = combine_text_fields(test_df)

    eda_summary = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_missing": train_df.isna().sum().to_dict(),
        "test_missing": test_df.isna().sum().to_dict(),
        "class_distribution": train_df["target"].value_counts().to_dict(),
        "class_distribution_normalized": train_df["target"].value_counts(normalize=True).round(4).to_dict(),
        "text_len_train": {
            "min": int(train_text.str.len().min()),
            "max": int(train_text.str.len().max()),
            "mean": float(train_text.str.len().mean()),
            "median": float(train_text.str.len().median()),
        },
        "text_len_test": {
            "min": int(test_text.str.len().min()),
            "max": int(test_text.str.len().max()),
            "mean": float(test_text.str.len().mean()),
            "median": float(test_text.str.len().median()),
        },
        "word_count_train": {
            "min": int(train_text.str.split().str.len().min()),
            "max": int(train_text.str.split().str.len().max()),
            "mean": float(train_text.str.split().str.len().mean()),
            "median": float(train_text.str.split().str.len().median()),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x="target", palette="Set2", hue="target", legend=False)
    plt.title("Target Class Distribution (Train)")
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_class_distribution.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(train_text.str.split().str.len(), bins=40, kde=True, color="#2a9d8f")
    plt.title("Train Text Word Count Distribution")
    plt.xlabel("words per sample")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_word_count_hist.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(train_text.str.len(), bins=40, kde=True, color="#e76f51")
    plt.title("Train Text Character Length Distribution")
    plt.xlabel("characters per sample")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_char_len_hist.png"), dpi=140)
    plt.close()

    return eda_summary


def build_model_tf(config: ModelConfig, max_words: int, max_len: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)

    model = tf.keras.Sequential(name=config.name)
    model.add(layers.Embedding(input_dim=max_words, output_dim=config.embedding_dim, input_length=max_len))

    if config.rnn_type == "SimpleRNN":
        model.add(
            layers.SimpleRNN(
                config.rnn_units,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout,
            )
        )
    elif config.rnn_type == "LSTM":
        model.add(
            layers.LSTM(
                config.rnn_units,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout,
            )
        )
    elif config.rnn_type == "GRU":
        model.add(
            layers.GRU(
                config.rnn_units,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout,
            )
        )
    else:
        raise ValueError(f"Unsupported rnn_type: {config.rnn_type}")

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def cross_validate_rnn(
    X_text: pd.Series,
    y: np.ndarray,
    configs: List[ModelConfig],
    n_splits: int = 5,
    max_words: int = 20000,
    max_len: int = 60,
) -> pd.DataFrame:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_rows = []

    for config in configs:
        print(f"\n=== CV for {config.name} ({config.rnn_type}) ===")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y), start=1):
            X_train_text = X_text.iloc[train_idx]
            X_val_text = X_text.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Fit tokenizer only on training fold text to avoid leakage.
            tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train_text.tolist())

            X_train_seq = tokenizer.texts_to_sequences(X_train_text.tolist())
            X_val_seq = tokenizer.texts_to_sequences(X_val_text.tolist())

            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
            X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding="post", truncating="post")

            model = build_model_tf(config, max_words=max_words, max_len=max_len)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]

            history = model.fit(
                X_train_pad,
                y_train,
                validation_data=(X_val_pad, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            y_prob = model.predict(X_val_pad, verbose=0).ravel()
            metrics = evaluate_metrics(y_val, y_prob)

            row = {
                "model": config.name,
                "rnn_type": config.rnn_type,
                "fold": fold,
                "best_val_loss": float(np.min(history.history["val_loss"])),
                "epochs_trained": int(len(history.history["loss"])),
                **metrics,
            }
            fold_rows.append(row)

            tf.keras.backend.clear_session()

            print(
                f"Fold {fold}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, "
                f"auc={metrics['roc_auc']:.4f}, best_val_loss={row['best_val_loss']:.4f}"
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv"), index=False)

    summary_df = (
        fold_df.groupby(["model", "rnn_type"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            best_val_loss_mean=("best_val_loss", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_std=("f1", "std"),
            roc_auc_std=("roc_auc", "std"),
        )
        .sort_values(["f1_mean", "accuracy_mean"], ascending=False)
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary_metrics.csv"), index=False)

    return summary_df


def train_best_and_predict(
    train_text: pd.Series,
    y: np.ndarray,
    test_text: pd.Series,
    best_config: ModelConfig,
    max_words: int = 20000,
    max_len: int = 60,
) -> Tuple[pd.DataFrame, Dict]:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text.tolist())

    X_train_seq = tokenizer.texts_to_sequences(train_text.tolist())
    X_test_seq = tokenizer.texts_to_sequences(test_text.tolist())

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

    model = build_model_tf(best_config, max_words=max_words, max_len=max_len)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train_pad,
        y,
        validation_split=0.1,
        epochs=best_config.epochs,
        batch_size=best_config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_prob = model.predict(X_test_pad, verbose=0).ravel()
    test_pred = (test_prob >= 0.5).astype(int)

    diagnostics = {
        "model_name": best_config.name,
        "rnn_type": best_config.rnn_type,
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
    }

    tf.keras.backend.clear_session()

    return pd.DataFrame({"target": test_pred, "target_probability": test_prob}), diagnostics


def write_report(eda_summary: Dict, cv_summary: pd.DataFrame, best_model_name: str, best_diag: Dict):
    report_path = os.path.join(OUTPUT_DIR, "model_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NLP Disaster Tweets RNN Assignment Report\n\n")
        f.write("## 1) EDA Summary\n")
        f.write(f"- Train shape: {tuple(eda_summary['train_shape'])}\n")
        f.write(f"- Test shape: {tuple(eda_summary['test_shape'])}\n")
        f.write(f"- Missing values (train): {eda_summary['train_missing']}\n")
        f.write(f"- Class distribution: {eda_summary['class_distribution']}\n")
        f.write(f"- Class ratio: {eda_summary['class_distribution_normalized']}\n")
        f.write(f"- Train text length stats: {eda_summary['text_len_train']}\n")
        f.write(f"- Train word count stats: {eda_summary['word_count_train']}\n\n")

        f.write("## 2) Cross-Validation Design\n")
        f.write("- StratifiedKFold with 5 folds and random_state=42\n")
        f.write("- Tokenizer fit only on each training fold (leakage control)\n")
        f.write("- Goodness-of-fit metrics: accuracy, precision, recall, F1, ROC-AUC, validation loss\n")
        f.write("- EarlyStopping(monitor=val_loss, patience=2, restore_best_weights=True)\n\n")

        f.write("## 3) Tuned RNN Models\n")
        for _, row in cv_summary.iterrows():
            f.write(
                f"- {row['model']} ({row['rnn_type']}): "
                f"F1={row['f1_mean']:.4f} (+/-{row['f1_std']:.4f}), "
                f"Acc={row['accuracy_mean']:.4f}, AUC={row['roc_auc_mean']:.4f}\n"
            )

        f.write("\n## 4) Best Model\n")
        f.write(f"- Selected by highest mean CV F1: **{best_model_name}**\n")
        f.write(f"- Final fit diagnostics on train/validation split: {best_diag}\n\n")

        f.write("## 5) Kaggle Submission\n")
        f.write("- Submission file: `outputs/submission_best_rnn.csv`\n")
        f.write("- Upload this file in Kaggle competition `nlp-getting-started`.\n")
        f.write("- Add your Kaggle username and screenshot(s) of score in your assignment write-up.\n")


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH)

    print("Running EDA...")
    eda_summary = run_eda(train_df, test_df)

    train_text = combine_text_fields(train_df)
    test_text = combine_text_fields(test_df)
    y = train_df["target"].values

    model_configs = [
        ModelConfig(
            name="rnn_simple_small",
            rnn_type="SimpleRNN",
            embedding_dim=64,
            rnn_units=64,
            dropout=0.3,
            recurrent_dropout=0.2,
            learning_rate=1e-3,
            batch_size=64,
            epochs=8,
        ),
        ModelConfig(
            name="rnn_lstm_medium",
            rnn_type="LSTM",
            embedding_dim=128,
            rnn_units=64,
            dropout=0.3,
            recurrent_dropout=0.2,
            learning_rate=8e-4,
            batch_size=64,
            epochs=10,
        ),
        ModelConfig(
            name="rnn_gru_medium",
            rnn_type="GRU",
            embedding_dim=128,
            rnn_units=96,
            dropout=0.3,
            recurrent_dropout=0.2,
            learning_rate=8e-4,
            batch_size=64,
            epochs=10,
        ),
    ]

    print("Starting cross-validation for 3 tuned RNN models...")
    try:
        cv_summary = cross_validate_rnn(
            X_text=train_text,
            y=y,
            configs=model_configs,
            n_splits=5,
            max_words=20000,
            max_len=60,
        )
    except ModuleNotFoundError as e:
        msg = (
            "TensorFlow is not installed in this environment. "
            "Install dependencies and rerun:\n"
            "  pip install tensorflow pandas numpy scikit-learn matplotlib seaborn\n"
        )
        print(msg)
        raise SystemExit(f"Missing dependency: {e}")

    print("\nCV summary:")
    print(cv_summary.to_string(index=False))

    best_model_name = cv_summary.iloc[0]["model"]
    best_config = next(c for c in model_configs if c.name == best_model_name)

    print(f"\nTraining best model on full train data: {best_model_name}")
    pred_df, best_diag = train_best_and_predict(
        train_text=train_text,
        y=y,
        test_text=test_text,
        best_config=best_config,
        max_words=20000,
        max_len=60,
    )

    submission = sample_sub_df.copy()
    submission["target"] = pred_df["target"].values
    submission_path = os.path.join(OUTPUT_DIR, "submission_best_rnn.csv")
    submission.to_csv(submission_path, index=False)

    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_prob.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "best_model_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(best_diag, f, indent=2)

    write_report(
        eda_summary=eda_summary,
        cv_summary=cv_summary,
        best_model_name=best_model_name,
        best_diag=best_diag,
    )

    print(f"\nDone. Outputs saved in: {OUTPUT_DIR}")
    print(f"Kaggle submission file: {submission_path}")


if __name__ == "__main__":
    main()
