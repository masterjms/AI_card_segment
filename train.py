import pandas as pd, numpy as np, joblib, os, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.impute import SimpleImputer
from pathlib import Path

# 경로
BASE_DIR = Path(f"./")
DATA_PATH = "C:/Users/mstot/PycharmProjects/PythonProject1/card_segment/data/preprocessing.csv"
MODEL_DIR = Path(f"./")
TARGET, ID_COL = "Segment.1", "ID"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 로드 (이미 인코딩·캐스팅 완료 상태)
df = pd.read_csv(DATA_PATH)

# 학습 피처 / 레이블
X_full = df.drop(columns=[TARGET, ID_COL]).astype(np.float32)  # float32 통일
# ── 숫자형만 선택 (문자형 변환 방지) ────────
X_full = X_full.select_dtypes(include=[np.number]).astype(np.float32)
y_full = df[TARGET].str.upper()  # 'A'~'E'

# (Stage, Positive Label, Remaining Labels) - A, B는 별도 처리
STAGES = [
    ("E", {"E"}),  # Stage-1 : E vs not-E
    ("D", {"D"}),  # Stage-2 : D vs not-D
    ("C", {"C"}),  # Stage-3 : C vs not-C
    # A, B는 별도 AB 전용 모델로 처리
]


def train_binary_stage(X, y, pos_label, *,
                       n_splits=5, random_state=42):
    # ── 1) 바이너리 라벨 생성 ───────────────────────────────
    y_bin = (y == pos_label).astype(int)  # 1 = positive
    pos_cnt = y_bin.sum()
    neg_cnt = len(y_bin) - pos_cnt

    # ── 2) scale_pos_weight : 항상 소수 클래스 > 1 ────────
    if pos_cnt == 0:  # 예외 방지
        spw = 1.0
    elif pos_cnt < neg_cnt:  # Positive 가 minority → 가중치 ↑
        spw = neg_cnt / pos_cnt
    else:  # Positive 가 majority → 가중치 1
        spw = 1.0

    params = dict(
        objective="binary:logistic",
        tree_method="hist",  # GPU
        device="cuda",
        max_bin=1024,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,  # ← 수정된 값
        eval_metric="aucpr",
        base_score=0.5,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    # OOF 버퍼
    oof_proba = np.zeros(len(X), dtype=np.float32)

    models, f1_scores = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y_bin)):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y_bin.iloc[tr])
        dvalid = xgb.DMatrix(X.iloc[va], label=y_bin.iloc[va])

        model = xgb.train(
            params, dtrain,
            num_boost_round=10_000,
            early_stopping_rounds=200,
            evals=[(dvalid, "val")],
            verbose_eval=300,
        )
        models.append(model)

        proba = model.predict(dvalid)
        oof_proba[va] = proba  # OOF 저장

        f1 = f1_score(y_bin.iloc[va],
                      (proba > 0.5).astype(int))
        f1_scores.append(f1)
        print(f"[Fold {fold}] {pos_label} F1 = {f1:.4f}")

    print(f"▶ {pos_label} Mean F1 = {np.mean(f1_scores):.4f}\n")
    return models, oof_proba, y_bin.values


# ─────────────────────────────────────────────
#  Precision 목표치 대폭 완화 (더 많은 데이터가 다음 단계로 넘어가도록)
# ─────────────────────────────────────────────
from sklearn.metrics import precision_recall_curve
import joblib, numpy as np, pandas as pd, os

#  Precision 목표치 대폭 완화 - Recall 확보를 위해
PREC_TARGET = {"E": 0.93,  # 0.998 → 0.95 (대폭 완화)
               "D": 0.90,  # 0.985 → 0.92 (대폭 완화)
               "C": 0.88,  # 0.970 → 0.90 (대폭 완화)
               }

stage_models = {}
thr_dict = {}
stage_stats = {}  # {label: TP·FP·…}
oof_buf = {}  # {label: {"idx":…, "proba":…}}

X_stage, y_stage = X_full, y_full.copy()
remain_idx_global = np.arange(len(df))  # 전체 DF 인덱스와 동기화

for label, _ in STAGES:
    print(f"\n========== Stage {label} ==========")

    # 1) 학습 & OOF 확률
    models, oof_p, y_bin = train_binary_stage(X_stage, y_stage, label)
    stage_models[label] = models
    joblib.dump(models, MODEL_DIR / f"stage_{label}.pkl")

    # 2) cut-off : Precision ≥ 목표치 중 Recall 최댓값 지점 (F1 fallback 일관 적용)
    p, r, th = precision_recall_curve(y_bin, oof_p)

    # precision·recall 의 첫 값(p=1, r=0) 제거 → 길이를 thresholds 와 맞춤
    p_aligned = p[1:]
    r_aligned = r[1:]

    mask_prec = p_aligned >= PREC_TARGET[label]

    if mask_prec.any():
        idx_best = r_aligned[mask_prec].argmax()
        best_thr = th[mask_prec][idx_best]
        print(f"   ↳ Precision 목표 달성: P={p_aligned[mask_prec][idx_best]:.4f}, R={r_aligned[mask_prec][idx_best]:.4f}")
    else:
        #  F1 fallback 일관 적용 - Precision 목표 미달 시
        f1_curve = 2 * p_aligned * r_aligned / (p_aligned + r_aligned + 1e-9)
        if len(f1_curve) > 0 and np.isfinite(f1_curve).any():
            best_idx = f1_curve.argmax()
            best_thr = th[best_idx]
            print(
                f"   ↳ F1 fallback 적용: P={p_aligned[best_idx]:.4f}, R={r_aligned[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}")
        else:
            best_thr = 0.5
            print(f"   ↳ 기본값 사용: thr=0.5")

    thr_dict[label] = float(best_thr)
    print(f"   ↳ best_thr({label}) = {best_thr:.4f}  "
          f"(target P={PREC_TARGET[label]:.3f})")

    # 3) 스테이지별 혼동행렬 집계
    pred_pos = oof_p > best_thr
    tp = int(((pred_pos) & (y_bin == 1)).sum())
    fp = int(((pred_pos) & (y_bin == 0)).sum())
    fn = int(((~pred_pos) & (y_bin == 1)).sum())
    tn = int(((~pred_pos) & (y_bin == 0)).sum())

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    stage_stats[label] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
                              precision=prec, recall=rec, F1=f1)

    # 4) OOF 확률 & 인덱스 저장  (2차 Cascade, 최종 OOF 평가용)
    oof_buf[label] = {"idx": remain_idx_global.copy(),
                      "proba": oof_p.astype(np.float32)}

    # 5) 다음 스테이지용 데이터 축소 (negative = not-Positive)
    mask_neg = ~pred_pos
    remain_idx_global = remain_idx_global[mask_neg]
    X_stage = X_stage.iloc[mask_neg]
    y_stage = y_stage.iloc[mask_neg]

    print(f"   ↳ 다음 단계로 넘어가는 샘플 수: {len(X_stage)}")

# 6) threshold 저장
joblib.dump(thr_dict, MODEL_DIR / "stage_thresholds.pkl")
joblib.dump(oof_buf, MODEL_DIR / "stage_oof_prob.pkl")
print("💾 모델 · threshold · OOF 확률 저장 완료")

# 7) 단계별 지표 출력
stats_df = (
    pd.DataFrame(stage_stats)
    .T[["TP", "FP", "FN", "TN", "precision", "recall", "F1"]]
    .round({"precision": 4, "recall": 4, "F1": 4})
)
print("\n=== Stage-wise Confusion / Metrics ===")
print(stats_df)

# ========================================================
#  A, B 전용 모델 (개선: 결측치 처리 + 오버샘플링)
# ========================================================
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import StratifiedKFold


def train_ab_specialized_model(X_full, y_full, random_state=42):
    """A, B만 대상으로 하는 전용 모델 (결측치 처리 + 오버샘플링 적용)"""
    print(f"\n========== A, B 전용 모델 학습 ==========")

    # A, B 클래스만 추출
    mask_ab = y_full.isin(['A', 'B'])
    X_ab = X_full[mask_ab].reset_index(drop=True)
    y_ab = y_full[mask_ab].reset_index(drop=True)

    #  전부 NaN인 컬럼 제거 (Imputer 에러 방지)
    X_ab = X_ab.dropna(axis=1, how='all')

    print(f"A, B 원본 분포:")
    print(y_ab.value_counts())

    # 결측치 처리 (SMOTE 오류 방지)
    print(f"결측치 처리 전 shape: {X_ab.shape}")
    print(f"결측치 개수: {X_ab.isnull().sum().sum()}")

    if X_ab.isnull().sum().sum() > 0:
        print("결측치 발견 - SimpleImputer로 처리")
        imputer = SimpleImputer(strategy='median')
        X_ab = pd.DataFrame(imputer.fit_transform(X_ab),
                            columns=X_ab.columns,
                            index=X_ab.index)
        print(f"결측치 처리 후: {X_ab.isnull().sum().sum()}개")

    # 오버샘플링 적용
    try:
        # k_neighbors를 안전한 값으로 설정
        min_class_size = min(y_ab.value_counts())
        k_neighbors = min(5, max(1, min_class_size - 1))

        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_ab_resampled, y_ab_resampled = smote.fit_resample(X_ab, y_ab)
        print(f"SMOTE 성공! k_neighbors={k_neighbors}")
        print(f"SMOTE 후 분포:")
        print(pd.Series(y_ab_resampled).value_counts())
    except Exception as e:
        print(f"SMOTE 실패: {e}")
        print("원본 데이터로 학습 진행")
        X_ab_resampled, y_ab_resampled = X_ab, y_ab

    # A vs B 바이너리 모델 학습 (A=1, B=0)
    y_ab_bin = (y_ab_resampled == 'A').astype(int)

    pos_cnt = y_ab_bin.sum()
    neg_cnt = len(y_ab_bin) - pos_cnt
    spw = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

    params = dict(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        max_bin=1024,
        learning_rate=0.05,  # 더 세밀한 학습
        max_depth=8,  # 더 깊은 트리
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        base_score=0.5,
        reg_alpha=0.1,  # L1 정규화
        reg_lambda=0.1,  # L2 정규화
    )

    # 원본 A, B 데이터에 대한 OOF 예측
    # CV splits 수 조정 (최소 클래스 크기 고려)
    n_splits = min(5, min(y_ab.value_counts()))
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba_ab = np.zeros(len(X_ab), dtype=np.float32)
    ab_models = []

    for fold, (tr, va) in enumerate(skf.split(X_ab, y_ab)):
        # 훈련 세트를 오버샘플링된 데이터로 사용
        dtrain = xgb.DMatrix(X_ab_resampled, label=y_ab_bin)
        dvalid = xgb.DMatrix(X_ab.iloc[va], label=(y_ab.iloc[va] == 'A').astype(int))

        model = xgb.train(
            params, dtrain,
            num_boost_round=3000,  # 샘플 수가 적으므로 줄임
            early_stopping_rounds=100,
            evals=[(dvalid, "val")],
            verbose_eval=200,
        )
        ab_models.append(model)

        proba = model.predict(dvalid)
        oof_proba_ab[va] = proba

        # Fold별 성능
        y_pred_fold = (proba > 0.5).astype(int)
        y_true_fold = (y_ab.iloc[va] == 'A').astype(int)
        f1_fold = f1_score(y_true_fold, y_pred_fold, zero_division=0)
        print(f"[AB Fold {fold}] F1 = {f1_fold:.4f}")

    #  Threshold 최적화 (F1 Score 기준 - 일관성 유지)
    y_ab_bin_orig = (y_ab == 'A').astype(int)

    # Precision-Recall curve 기반 threshold 계산
    try:
        p, r, th = precision_recall_curve(y_ab_bin_orig, oof_proba_ab)
        p_aligned = p[1:]
        r_aligned = r[1:]

        # F1 최대화 방식 사용 (AB 모델은 특별히 precision 타겟 없음)
        f1_curve = 2 * p_aligned * r_aligned / (p_aligned + r_aligned + 1e-9)
        if len(f1_curve) > 0 and np.isfinite(f1_curve).any():
            best_idx = f1_curve.argmax()
            best_thr_ab = th[best_idx]
            print(f"   ↳ AB 모델 F1 기준 최적 threshold = {best_thr_ab:.4f}")
            print(f"   ↳ 최적점: P={p_aligned[best_idx]:.4f}, R={r_aligned[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}")
        else:
            best_thr_ab = 0.5
            print(f"   ↳ AB 모델 기본값 사용: thr=0.5")
    except Exception as e:
        print(f"   ↳ AB threshold 계산 오류: {e}, 기본값 사용")
        best_thr_ab = 0.5

    # 성능 평가
    pred_ab = (oof_proba_ab > best_thr_ab).astype(int)
    tp_ab = int(((pred_ab == 1) & (y_ab_bin_orig == 1)).sum())
    fp_ab = int(((pred_ab == 1) & (y_ab_bin_orig == 0)).sum())
    fn_ab = int(((pred_ab == 0) & (y_ab_bin_orig == 1)).sum())
    tn_ab = int(((pred_ab == 0) & (y_ab_bin_orig == 0)).sum())

    prec_ab = tp_ab / (tp_ab + fp_ab + 1e-9)
    rec_ab = tp_ab / (tp_ab + fn_ab + 1e-9)
    f1_ab = 2 * prec_ab * rec_ab / (prec_ab + rec_ab + 1e-9)

    ab_stats = dict(TP=tp_ab, FP=fp_ab, FN=fn_ab, TN=tn_ab,
                    precision=prec_ab, recall=rec_ab, F1=f1_ab)

    # 원본 인덱스 복원
    ab_indices = np.where(mask_ab)[0]

    return ab_models, best_thr_ab, ab_stats, {
        "idx": ab_indices,
        "proba": oof_proba_ab.astype(np.float32)
    }


# A, B 전용 모델 학습
ab_models, ab_threshold, ab_stats, ab_oof_buf = train_ab_specialized_model(X_full, y_full)

# A, B 모델 저장
joblib.dump(ab_models, MODEL_DIR / "ab_specialized_model.pkl")
joblib.dump({"threshold": ab_threshold}, MODEL_DIR / "ab_threshold.pkl")
joblib.dump(ab_oof_buf, MODEL_DIR / "ab_oof_prob.pkl")

print(f"\n=== A, B 전용 모델 성능 ===")
print(f"Precision: {ab_stats['precision']:.4f}")
print(f"Recall: {ab_stats['recall']:.4f}")
print(f"F1: {ab_stats['F1']:.4f}")


# ───────────────────────  2차 Cascade (E, D, C만)  ─────────────────────

# LightGBM 학습 함수 정의 (특수문자 오류 해결 + F1 fallback 일관 적용)
def train_lgb_stage(X, y, pos_label, *, n_splits=5, random_state=42):
    """LightGBM 기반 바이너리 분류 학습 (특수문자 오류 해결 + F1 fallback 일관 적용)"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("⚠️ LightGBM not installed. 설치 필요: pip install lightgbm")
        return [], np.zeros(len(X), dtype=np.float32), 0.5, np.zeros(len(X))

    from sklearn.metrics import precision_recall_curve

    # 🔥 특수문자 오류 해결: 컬럼명 정리
    X_clean = X.copy()
    X_clean.columns = X_clean.columns.str.replace(r"[^\w]", "_", regex=True)
    X_clean.columns = X_clean.columns.str.replace(r"_+", "_", regex=True)
    X_clean.columns = X_clean.columns.str.strip("_")

    print(f"   ↳ 컬럼명 정리 완료: {X.shape[1]}개 피처")

    # 바이너리 라벨 생성
    y_bin = (y == pos_label).astype(int)
    pos_cnt = y_bin.sum()
    neg_cnt = len(y_bin) - pos_cnt

    if pos_cnt == 0:
        print(f"⚠️ {pos_label}: 양성 샘플 없음, 빈 모델 반환")
        return [], np.zeros(len(X), dtype=np.float32), 0.5, y_bin.values

    # 최소 샘플 수 체크
    if len(X) < 10:
        print(f"⚠️ {pos_label}: 샘플 수가 너무 적음 ({len(X)}개), 빈 모델 반환")
        return [], np.zeros(len(X), dtype=np.float32), 0.5, y_bin.values

    # CV splits 조정
    actual_splits = min(n_splits, min(pos_cnt, neg_cnt))
    if actual_splits < 2:
        actual_splits = 2
        print(f"⚠️ {pos_label}: CV splits를 {actual_splits}로 조정")

    # scale_pos_weight 계산
    spw = neg_cnt / pos_cnt if pos_cnt < neg_cnt else 1.0

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': min(31, max(10, len(X) // 20)),
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': spw,
        'verbose': -1,
        'random_state': random_state,
        'force_col_wise': True,
        'min_data_in_leaf': max(1, len(X) // 100)
    }

    try:
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        oof_proba = np.zeros(len(X), dtype=np.float32)
        models = []

        for fold, (tr, va) in enumerate(skf.split(X_clean, y_bin)):
            # 데이터 검증
            if len(tr) == 0 or len(va) == 0:
                print(f"⚠️ {pos_label} Fold {fold}: 빈 훈련/검증 세트")
                continue

            # 각 fold에서 클래스 균형 확인
            if y_bin.iloc[tr].sum() == 0 or y_bin.iloc[va].sum() == 0:
                print(f"⚠️ {pos_label} Fold {fold}: 한쪽 클래스만 존재")
                oof_proba[va] = 0.5
                continue

            train_data = lgb.Dataset(
                X_clean.iloc[tr],
                label=y_bin.iloc[tr],
                free_raw_data=False
            )
            valid_data = lgb.Dataset(
                X_clean.iloc[va],
                label=y_bin.iloc[va],
                reference=train_data,
                free_raw_data=False
            )

            early_stopping_rounds = min(100, max(10, 1000 // actual_splits))

            model = lgb.train(
                params, train_data,
                num_boost_round=1000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0)
                ]
            )
            models.append(model)

            # 예측 시 안전장치
            try:
                proba = model.predict(X_clean.iloc[va], num_iteration=model.best_iteration)
                proba = np.where(np.isfinite(proba), proba, 0.5)
                proba = np.clip(proba, 1e-7, 1 - 1e-7)
                oof_proba[va] = proba
            except Exception as e:
                print(f"⚠️ {pos_label} Fold {fold} 예측 오류: {e}")
                oof_proba[va] = 0.5

        if len(models) == 0:
            print(f"⚠️ {pos_label}: 학습된 모델이 없음")
            return [], np.full(len(X), 0.5, dtype=np.float32), 0.5, y_bin.values

        #  Threshold 계산 (Precision 기준 + F1 fallback 일관 적용)
        prec_target = PREC_TGT2.get(pos_label, 0.9)

        try:
            p, r, th = precision_recall_curve(y_bin, oof_proba)
            p_aligned, r_aligned = p[1:], r[1:]

            mask_prec = p_aligned >= prec_target
            if mask_prec.any():
                best_idx = r_aligned[mask_prec].argmax()
                best_thr = th[mask_prec][best_idx]
                print(
                    f"   ↳ LGB {pos_label}: Precision 목표 달성 - P={p_aligned[mask_prec][best_idx]:.4f}, R={r_aligned[mask_prec][best_idx]:.4f}")
            else:
                #  F1 fallback 일관 적용
                f1_curve = 2 * p_aligned * r_aligned / (p_aligned + r_aligned + 1e-9)
                if len(f1_curve) > 0 and np.isfinite(f1_curve).any():
                    best_idx = f1_curve.argmax()
                    best_thr = th[best_idx]
                    print(
                        f"   ↳ LGB {pos_label}: F1 fallback 적용 - P={p_aligned[best_idx]:.4f}, R={r_aligned[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}")
                else:
                    best_thr = 0.5
                    print(f"   ↳ LGB {pos_label}: 기본값 사용 - thr=0.5")
        except Exception as e:
            print(f"⚠️ {pos_label}: Threshold 계산 오류: {e}")
            best_thr = 0.5

        print(f"   ↳ LGB {pos_label}: thr={best_thr:.4f}, target_P={prec_target:.3f}")
        return models, oof_proba, best_thr, y_bin.values

    except Exception as e:
        print(f"⚠️ {pos_label}: LightGBM 학습 전체 오류: {e}")
        return [], np.full(len(X), 0.5, dtype=np.float32), 0.5, np.zeros(len(X))


##  2차 Cascade용 Precision 목표치 (더 엄격하게 설정)
PREC_TGT2 = {"E": 0.90,  # 2차에서는 더 높은 정밀도
             "D": 0.85,
             "C": 0.83}

# 2차 Cascade 모델 학습 (LightGBM)
stage_models_2nd = {}
thr_dict_2nd = {}
stage_stats_2nd = {}
oof_buf_2nd = {}

# 1차에서 negative로 분류된 샘플들만 사용
print(f"\n========== 2차 Cascade 학습 시작 ==========")

# 1차에서 남은 샘플들 (E, D, C가 모두 negative로 분류된 샘플들)
# 전체 데이터에서 1차 스테이지별로 positive 예측된 샘플들 제외
excluded_indices = set()

for label in ["E", "D", "C"]:
    stage_oof = oof_buf[label]
    stage_thr = thr_dict[label]
    pos_mask = stage_oof["proba"] > stage_thr
    pos_indices = stage_oof["idx"][pos_mask]
    excluded_indices.update(pos_indices)

# 2차 학습용 데이터 (1차에서 모든 단계를 통과한 샘플들)
remaining_indices = np.array([i for i in range(len(df)) if i not in excluded_indices])
X_2nd = X_full.iloc[remaining_indices].reset_index(drop=True)
y_2nd = y_full.iloc[remaining_indices].reset_index(drop=True)

print(f"2차 Cascade 학습 데이터: {len(X_2nd)}개 샘플")
print(f"2차 레이블 분포:\n{y_2nd.value_counts()}")

# 2차 Cascade 단계별 학습
X_stage_2nd, y_stage_2nd = X_2nd, y_2nd.copy()
remain_idx_2nd = remaining_indices.copy()

for label, _ in STAGES:
    if len(X_stage_2nd) == 0:
        print(f"⚠️ {label} 2차: 학습할 데이터가 없습니다.")
        continue

    print(f"\n---------- 2차 {label} 단계 ----------")

    # LightGBM으로 학습
    models_2nd, oof_p_2nd, best_thr_2nd, y_bin_2nd = train_lgb_stage(
        X_stage_2nd, y_stage_2nd, label
    )

    if len(models_2nd) == 0:
        print(f"⚠️ {label} 2차: 모델 학습 실패")
        continue

    stage_models_2nd[label] = models_2nd
    thr_dict_2nd[label] = float(best_thr_2nd)

    # 성능 평가
    pred_pos_2nd = oof_p_2nd > best_thr_2nd
    tp_2nd = int(((pred_pos_2nd) & (y_bin_2nd == 1)).sum())
    fp_2nd = int(((pred_pos_2nd) & (y_bin_2nd == 0)).sum())
    fn_2nd = int(((~pred_pos_2nd) & (y_bin_2nd == 1)).sum())
    tn_2nd = int(((~pred_pos_2nd) & (y_bin_2nd == 0)).sum())

    prec_2nd = tp_2nd / (tp_2nd + fp_2nd + 1e-9)
    rec_2nd = tp_2nd / (tp_2nd + fn_2nd + 1e-9)
    f1_2nd = 2 * prec_2nd * rec_2nd / (prec_2nd + rec_2nd + 1e-9)

    stage_stats_2nd[label] = dict(TP=tp_2nd, FP=fp_2nd, FN=fn_2nd, TN=tn_2nd,
                                  precision=prec_2nd, recall=rec_2nd, F1=f1_2nd)

    # OOF 저장
    oof_buf_2nd[label] = {"idx": remain_idx_2nd.copy(),
                          "proba": oof_p_2nd.astype(np.float32)}

    # 다음 단계용 데이터 축소
    mask_neg_2nd = ~pred_pos_2nd
    remain_idx_2nd = remain_idx_2nd[mask_neg_2nd]
    X_stage_2nd = X_stage_2nd.iloc[mask_neg_2nd].reset_index(drop=True)
    y_stage_2nd = y_stage_2nd.iloc[mask_neg_2nd].reset_index(drop=True)

    print(f"   ↳ 2차 {label} 완료: P={prec_2nd:.4f}, R={rec_2nd:.4f}, F1={f1_2nd:.4f}")
    print(f"   ↳ 다음 단계 샘플 수: {len(X_stage_2nd)}")

# 2차 모델 저장
joblib.dump(stage_models_2nd, MODEL_DIR / "stage_models_2nd.pkl")
joblib.dump(thr_dict_2nd, MODEL_DIR / "stage_thresholds_2nd.pkl")
joblib.dump(oof_buf_2nd, MODEL_DIR / "stage_oof_prob_2nd.pkl")

# 2차 결과 출력
if stage_stats_2nd:
    stats_df_2nd = (
        pd.DataFrame(stage_stats_2nd)
        .T[["TP", "FP", "FN", "TN", "precision", "recall", "F1"]]
        .round({"precision": 4, "recall": 4, "F1": 4})
    )
    print("\n=== 2차 Cascade Stage-wise Metrics ===")
    print(stats_df_2nd)

# ============================================================
#  최종 OOF 예측 결과 생성 (1차 + 2차 + AB 결합)
# ============================================================

print(f"\n========== 최종 OOF 예측 결과 생성 ==========")

# 전체 데이터에 대한 최종 예측 초기화
final_predictions = pd.Series(['UNKNOWN'] * len(df), index=df.index)

# 1) 1차 Cascade 결과 적용
for label in ["E", "D", "C"]:
    if label in oof_buf:
        stage_oof = oof_buf[label]
        stage_thr = thr_dict[label]
        pos_mask = stage_oof["proba"] > stage_thr
        pos_indices = stage_oof["idx"][pos_mask]
        final_predictions.iloc[pos_indices] = label
        print(f"1차 {label}: {len(pos_indices)}개 샘플 예측")

# 2) 2차 Cascade 결과 적용 (1차에서 예측되지 않은 샘플들 중에서)
for label in ["E", "D", "C"]:
    if label in oof_buf_2nd:
        stage_oof_2nd = oof_buf_2nd[label]
        stage_thr_2nd = thr_dict_2nd[label]
        pos_mask_2nd = stage_oof_2nd["proba"] > stage_thr_2nd
        pos_indices_2nd = stage_oof_2nd["idx"][pos_mask_2nd]

        # 아직 예측되지 않은 샘플들만 업데이트
        unassigned_mask = final_predictions.iloc[pos_indices_2nd] == 'UNKNOWN'
        final_pos_indices = pos_indices_2nd[unassigned_mask]
        final_predictions.iloc[final_pos_indices] = label
        print(f"2차 {label}: {len(final_pos_indices)}개 추가 샘플 예측")

# 3) A, B 전용 모델 결과 적용
if ab_oof_buf and ab_threshold:
    ab_indices = ab_oof_buf["idx"]
    ab_proba = ab_oof_buf["proba"]

    # A vs B 예측 (A=1, B=0)
    pred_A = ab_proba > ab_threshold
    pred_B = ~pred_A

    final_predictions.iloc[ab_indices[pred_A]] = 'A'
    final_predictions.iloc[ab_indices[pred_B]] = 'B'
    print(f"AB 모델: A={pred_A.sum()}개, B={pred_B.sum()}개 샘플 예측")

# 4) 예측되지 않은 샘플들 처리 (기본값 할당)
unassigned_mask = final_predictions == 'UNKNOWN'
unassigned_count = unassigned_mask.sum()

if unassigned_count > 0:
    print(f"⚠️ 예측되지 않은 샘플: {unassigned_count}개")

    # 실제 레이블 분포 기반으로 기본값 할당
    actual_dist = y_full.value_counts(normalize=True)
    most_common_label = actual_dist.index[0]

    final_predictions[unassigned_mask] = most_common_label
    print(f"   ↳ 기본값 '{most_common_label}' 할당")

# 5) 최종 결과 평가
print(f"\n=== 최종 OOF 예측 결과 ===")
print(f"예측 분포:")
print(final_predictions.value_counts())

print(f"\n실제 분포:")
print(y_full.value_counts())

# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

# 1. 문자열 캐스팅
y_true = y_full.astype(str)
y_pred = final_predictions.astype(str)

# 2. classification report
print("\n=== 최종 Classification Report ===")
print(classification_report(y_true, y_pred, zero_division=0))

# 3. confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
cm_df = pd.DataFrame(cm,
                     index=[f"True_{l}" for l in sorted(y_true.unique())],
                     columns=[f"Pred_{l}" for l in sorted(y_true.unique())])
print("\n=== 최종 Confusion Matrix ===")
print(cm_df)

# print(f"\n=== 최종 Classification Report ===")
# print(classification_report(y_full, final_predictions, zero_division=0))
#
# # Confusion Matrix
# print(f"\n=== 최종 Confusion Matrix ===")
# cm = confusion_matrix(y_full, final_predictions)
# cm_df = pd.DataFrame(cm,
#                      index=[f"True_{label}" for label in sorted(y_full.unique())],
#                      columns=[f"Pred_{label}" for label in sorted(final_predictions.unique())])
# print(cm_df)

# 전체 정확도
overall_accuracy = (final_predictions == y_full).mean()
print(f"\n전체 정확도: {overall_accuracy:.4f}")

# 클래스별 정확도
class_accuracy = {}
for label in y_full.unique():
    mask = y_full == label
    if mask.sum() > 0:
        acc = (final_predictions[mask] == label).mean()
        class_accuracy[label] = acc

print(f"\n클래스별 정확도:")
for label, acc in sorted(class_accuracy.items()):
    print(f"  {label}: {acc:.4f}")

# 최종 결과 저장
result_df = pd.DataFrame({
    'ID': df[ID_COL],
    'Actual': y_full,
    'Predicted': final_predictions
})
result_df.to_csv(MODEL_DIR / "final_oof_predictions.csv", index=False)

print(f"\n💾 최종 결과 저장 완료: {MODEL_DIR / 'final_oof_predictions.csv'}")
print(f"🎉 전체 파이프라인 완료!")

# ============================================================
# 🔥 모델 성능 요약
# ============================================================

print(f"\n" + "=" * 60)
print(f"📊 모델 성능 요약")
print(f"=" * 60)

print(f"\n 1차 Cascade (XGBoost)")
if not stats_df.empty:
    print(stats_df)

print(f"\n 2차 Cascade (LightGBM)")
if 'stats_df_2nd' in locals() and not stats_df_2nd.empty:
    print(stats_df_2nd)

print(f"\n A, B 전용 모델")
print(f"Precision: {ab_stats['precision']:.4f}")
print(f"Recall: {ab_stats['recall']:.4f}")
print(f"F1: {ab_stats['F1']:.4f}")

print(f"\n 최종 통합 성능")
print(f"전체 정확도: {overall_accuracy:.4f}")
print(f"클래스별 정확도: {dict(sorted(class_accuracy.items()))}")

print(f"\n 저장된 파일들:")
saved_files = [
    "stage_E.pkl", "stage_D.pkl", "stage_C.pkl",
    "stage_thresholds.pkl", "stage_oof_prob.pkl",
    "ab_specialized_model.pkl", "ab_threshold.pkl", "ab_oof_prob.pkl",
    "stage_models_2nd.pkl", "stage_thresholds_2nd.pkl", "stage_oof_prob_2nd.pkl",
    "final_oof_predictions.csv"
]

for file in saved_files:
    file_path = MODEL_DIR / file
    if file_path.exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (누락)")

print(f"\n🎯 완료! 모든 모델과 결과가 저장되었습니다.")

