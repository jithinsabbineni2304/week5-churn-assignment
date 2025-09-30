import pandas as pd
from pathlib import Path

try:
    from pycaret.classification import load_model, predict_model
except Exception as e:
    raise ImportError("PyCaret is required to use this module. Please install pycaret>=3.x") from e

def _try_load_model(model_path_base: str):
    p = Path(model_path_base)
    if p.suffix.lower() != ".pkl":
        # Try base name, then add .pkl
        try:
            return load_model(model_path_base)
        except Exception:
            return load_model(str(p.with_suffix(".pkl")))
    else:
        return load_model(str(p))

def predict_proba_df(df_or_path, model_path_base="week5_churn_model"):
    if isinstance(df_or_path, (str, Path)):
        df_infer = pd.read_csv(df_or_path)
    else:
        df_infer = df_or_path.copy()

    model = _try_load_model(model_path_base)
    preds = predict_model(model, data=df_infer, raw_score=True)  # raw_score exposes probabilities when available
    proba_cols = [c for c in preds.columns if c.lower().startswith("score") or c.lower().startswith("prob")]
    if not proba_cols:
        proba_cols = [c for c in preds.columns if "probability" in c.lower()]
    return preds, proba_cols

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python churn_predictor.py <csv_path> [model_base]")
        sys.exit(1)
    csv_path = sys.argv[1]
    model_base = sys.argv[2] if len(sys.argv) > 2 else "week5_churn_model"
    out_df, proba_cols = predict_proba_df(csv_path, model_path_base=model_base)
    if proba_cols:
        print(out_df[proba_cols].to_string(index=False))
    else:
        print("Predictions (no explicit probability columns found):")
        print(out_df.head().to_string(index=False))
