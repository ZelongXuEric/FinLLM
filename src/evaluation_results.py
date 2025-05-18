import os
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

try:
    BASE = Path(os.environ.get("FINLLM_BASE", Path(__file__).resolve().parent.parent))
except NameError:
    BASE = Path(os.environ.get("FINLLM_BASE", "."))

RESULT_DIR = BASE / "results"

KEY_CAUSAL = "causal_llm_pred"
KEY_ASSOC  = "assoc_llm_pred"

def calculate_and_print_metrics(df_results: pd.DataFrame, model_name: str):
    """
    Calculates and prints binary classification metrics for causal and association predictions.
    Now includes model_name in the output.
    """
    print(f"\n===== Metrics for Model: {model_name} =====")

    # Evaluate Causal Prediction
    gt_causal_col = "ground_causal"
    pred_causal_col = KEY_CAUSAL
    valid_causal = df_results[
        (df_results[gt_causal_col].astype(str).isin(["Yes", "No"])) &
        (df_results[pred_causal_col].astype(str).isin(["Yes", "No"]))
    ]
    if not valid_causal.empty:
        acc = accuracy_score(valid_causal[gt_causal_col], valid_causal[pred_causal_col])
        p, r, f, _ = precision_recall_fscore_support(
            valid_causal[gt_causal_col],
            valid_causal[pred_causal_col],
            pos_label="Yes",
            average="binary",
            zero_division=0
        )
        print(f"Causal Prediction ({pred_causal_col}):")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Precision (for 'Yes'): {p:.2%}")
        print(f"  Recall (for 'Yes'): {r:.2%}")
        print(f"  F1-score (for 'Yes'): {f:.2%}")
        print(f"  (Evaluated on {len(valid_causal)} valid samples)")
    else:
        logging.warning(f"No valid samples for causal prediction for model {model_name}.")
        print(f"Causal Prediction ({pred_causal_col}): No valid samples to evaluate.")

    print("-" * 20)

    # Evaluate Association Prediction
    gt_assoc_col = "ground_assoc"
    pred_assoc_col = KEY_ASSOC
    valid_assoc = df_results[
        (df_results[gt_assoc_col].astype(str).isin(["Yes", "No"])) &
        (df_results[pred_assoc_col].astype(str).isin(["Yes", "No"]))
    ]
    if not valid_assoc.empty:
        acc = accuracy_score(valid_assoc[gt_assoc_col], valid_assoc[pred_assoc_col])
        p, r, f, _ = precision_recall_fscore_support(
            valid_assoc[gt_assoc_col],
            valid_assoc[pred_assoc_col],
            pos_label="Yes",
            average="binary",
            zero_division=0
        )
        print(f"Association Prediction ({pred_assoc_col}):")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Precision (for 'Yes'): {p:.2%}")
        print(f"  Recall (for 'Yes'): {r:.2%}")
        print(f"  F1-score (for 'Yes'): {f:.2%}")
        print(f"  (Evaluated on {len(valid_assoc)} valid samples)")
    else:
        logging.warning(f"No valid samples for association prediction for model {model_name}.")
        print(f"Association Prediction ({pred_assoc_col}): No valid samples to evaluate.")
    print("=" * 35)


if __name__ == "__main__":
    if not RESULT_DIR.exists():
        logging.error(f"Results directory not found: {RESULT_DIR}")
        exit(1)

    result_files = sorted(list(RESULT_DIR.glob("*_benchmark.csv")))

    if not result_files:
        logging.info(f"No benchmark result files found in {RESULT_DIR}")
        exit(0)

    logging.info(f"Found {len(result_files)} benchmark result files. Processing...")

    for result_file_path in result_files:
        model_name_from_file = result_file_path.name.replace("_benchmark.csv", "")
        
        try:
            df = pd.read_csv(result_file_path)
            if df.empty:
                logging.warning(f"Result file {result_file_path} is empty. Skipping.")
                continue

            # Ensure relevant columns are treated as strings for .isin and comparison
            for col in ["ground_causal", "ground_assoc", KEY_CAUSAL, KEY_ASSOC]:
                if col in df.columns:
                    df[col] = df[col].astype(str) 
                else: 
                    logging.warning(f"Column '{col}' not found in {result_file_path}. Skipping its metrics or defaulting.")


            calculate_and_print_metrics(df, model_name_from_file)
        except Exception as e:
            logging.error(f"Error processing file {result_file_path}: {e}")

    logging.info("Evaluation of all benchmark results complete.")