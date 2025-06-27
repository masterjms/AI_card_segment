# infer.py
import pandas as pd, numpy as np, joblib, xgboost as xgb, lightgbm as lgb
from pathlib import Path
from catboost import CatBoostClassifier

# ── 경로 설정 ─────────────────────────────────────────────
BASE_DIR = Path(f"./")
MODEL_DIR = Path(f"./")
TEST_PATH = "C:/Users/mstot/PycharmProjects/PythonProject1/card_segment/data/preprocessing.csv"
SUB_PATH = BASE_DIR / "submission.csv"

TARGET, ID_COL = "Segment.1", "ID"
STAGES = ["E", "D", "C"]

# ── 모델 및 threshold 로드 ────────────────────────────────
print("▶ 모델 로드")
stage_models_1st = {k: joblib.load(MODEL_DIR / f"stage_{k}.pkl") for k in STAGES}
thr_dict_1st = joblib.load(MODEL_DIR / "stage_thresholds.pkl")
ab_models = joblib.load(MODEL_DIR / "ab_specialized_model.pkl")
ab_thr = joblib.load(MODEL_DIR / "ab_threshold.pkl")["threshold"]
stage_models_2nd = joblib.load(MODEL_DIR / "stage_models_2nd.pkl")
thr_dict_2nd = joblib.load(MODEL_DIR / "stage_thresholds_2nd.pkl")

# ── 테스트 데이터 로드 ─────────────────────────────────────
print("▶ 테스트 로드")
df_test = pd.read_parquet(TEST_PATH)
X_test = df_test.drop(columns=[ID_COL])
X_test = X_test.select_dtypes(include=[np.number]).astype(np.float32)
test_indices = df_test.index
y_pred = pd.Series(["UNKNOWN"] * len(df_test), index=test_indices)

# ── 1차 Cascade: XGBoost 예측 ─────────────────────────────
for label in STAGES:
    models = stage_models_1st[label]
    thr = thr_dict_1st[label]
    proba = np.mean([m.predict(xgb.DMatrix(X_test)) for m in models], axis=0)
    idx = (proba > thr) & (y_pred == "UNKNOWN")
    y_pred.loc[idx] = label
    print(f"[1차 {label}] 예측된 샘플 수: {idx.sum()}")

# ── 2차 Cascade: LightGBM 예측 ─────────────────────────────
for label in STAGES:
    models = stage_models_2nd.get(label, [])
    if not models:
        continue
    thr = thr_dict_2nd[label]
    proba = np.mean([m.predict(X_test) for m in models], axis=0)
    idx = (proba > thr) & (y_pred == "UNKNOWN")
    y_pred.loc[idx] = label
    print(f"[2차 {label}] 예측된 샘플 수: {idx.sum()}")

# ── A/B 전용 모델 예측 ─────────────────────────────────────
ab_idx = y_pred[y_pred == "UNKNOWN"].index
if len(ab_idx) > 0:
    X_ab = X_test.loc[ab_idx]
    proba_ab = np.mean([m.predict(xgb.DMatrix(X_ab)) for m in ab_models], axis=0)
    pred_ab = np.where(proba_ab > ab_thr, "A", "B")
    y_pred.loc[ab_idx] = pred_ab
    print(f"[AB 모델] A={np.sum(pred_ab == 'A')}, B={np.sum(pred_ab == 'B')}")

# ── 최종 확인 및 저장 ─────────────────────────────────────
print(f"최종 예측 분포:\n{y_pred.value_counts()}")
submission = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET: y_pred})
submission.to_csv(SUB_PATH, index=False)
print(f"💾 저장 완료: {SUB_PATH}")
