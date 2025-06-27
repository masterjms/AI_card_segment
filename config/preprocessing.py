import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# 파일 경로
input_path = "./data/card_merged.csv"
output_path = "./data/Final.csv"

# 데이터 로드
df = pd.read_csv(input_path, low_memory=False)

# 고객 ID 생성
if "고객ID" not in df.columns:
    df["고객ID"] = df.index.astype(str)

# -------------------------------------------
# 🔻 1. 불필요한 열 제거 전, 종합방문횟수 먼저 계산
# -------------------------------------------
visit_cols = [
    "방문횟수_PC_R6M", "방문일수_PC_R6M", "방문횟수_앱_R6M",
    "방문일수_앱_R6M", "방문횟수_모바일웹_R6M", "방문일수_모바일웹_R6M",
    "방문횟수_PC_B0M", "방문일수_PC_B0M", "방문횟수_앱_B0M", "방문일수_앱_B0M",
    "방문횟수_모바일웹_B0M", "방문일수_모바일웹_B0M"
]
existing_visits = [col for col in visit_cols if col in df.columns]
df["종합방문횟수"] = df[existing_visits].sum(axis=1, skipna=True)

# -------------------------------------------
# 🔻 2. 불필요한 열 제거
# -------------------------------------------
drop_cols = [
    "Unnamed: 0.1", "대표결제일", "통합_납부_업종", "연체일자_B0M",
    "_1순위업종", "_2순위업종", "_3순위업종",
    "_1순위쇼핑업종", "_2순위쇼핑업종", "_3순위쇼핑업종",
    "_1순위교통업종", "_2순위교통업종", "_3순위교통업종",
    "_1순위여유업종", "_2순위여유업종", "_3순위여유업종",
    "_1순위납부업종", "_2순위납부업종", "_3순위납부업종",
    "통합_기본업종", "통합_쇼핑업종", "통합_교통업종", "통합_여유업종", "통합_납부업종",
    "상품관련면제카드수_B0M", "캠페인접촉일수_R12M", "시장단기연체여부_R6M"
] + existing_visits
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# -------------------------------------------
# 🔸 3. 이상치 처리
# -------------------------------------------
replace_vals = [-9999999, -99, 999999999, 999]
df.replace(replace_vals, np.nan, inplace=True)

if 'rv최초시작후경과일' in df.columns:
    df['rv최초시작후경과일'] = df['rv최초시작후경과일'].replace(99999999, np.nan)

# -------------------------------------------
# 🔸 4. 문자형 숫자 변환
# -------------------------------------------
def parse_text_to_number(x):
    if pd.isna(x): return np.nan
    x = str(x).replace("회이상", "").replace("회 이상", "").replace("회", "")
    x = x.replace("만원", "").replace(",", "").replace(" ", "")
    x = x.replace("이상", "").replace("세", "").replace("대", "")
    try: return float(x)
    except: return np.nan

parse_cols = [
    '할인건수_R3M', '이용메뉴건수_ARS_R6M', '방문횟수_PC_R6M',
    '방문일수_PC_R6M', '방문횟수_앱_R6M', '캡페인접촉건수_R12M', '캠페인접촉건수_R6M'
]
for col in parse_cols:
    if col in df.columns:
        df[col] = df[col].map(parse_text_to_number)

# -------------------------------------------
# 🔸 5. 파생변수 생성: 모멘텀갭
# -------------------------------------------
req_cols = [
    "연체입금원금_B0M", "연체입금원금_B2M", "연체입금원금_B5M",
    "정상청구원금_B0M", "정상청구원금_B2M", "정상청구원금_B5M"
]
if all(col in df.columns for col in req_cols):
    df["모멘텀갭_B0M_B2M"] = (df["연체입금원금_B0M"] - df["연체입금원금_B2M"]) - \
                            (df["정상청구원금_B0M"] - df["정상청구원금_B2M"])
    df["모멘텀갭_B2M_B5M"] = (df["연체입금원금_B2M"] - df["연체입금원금_B5M"]) - \
                            (df["정상청구원금_B2M"] - df["정상청구원금_B5M"])
    df["모멘텀갭_B0M_B5M"] = (df["연체입금원금_B0M"] - df["연체입금원금_B5M"]) - \
                            (df["정상청구원금_B0M"] - df["정상청구원금_B5M"])
    for col in ["모멘텀갭_B0M_B2M", "모멘텀갭_B2M_B5M", "모멘텀갭_B0M_B5M"]:
        df[f"{col}_log"] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

# -------------------------------------------
# 🔸 6. 범주형 라벨 인코딩
# -------------------------------------------
label_cols = [
    '대표결제방법코드', '가입통신회사코드', '_1순위신용체크구분', '_2순위신용체크구분',
    'RV전환가능여부', 'OS구분코드', 'Segment', 'Life_Stage'
]
label_mappings = {}
for col in label_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = le.classes_

# -------------------------------------------
# 🔸 7. 결측치 처리
# -------------------------------------------
if 'rv최초시작후경과일' in df.columns:
    df['rv최초시작후경과일'] = df['rv최초시작후경과일'].fillna(0)

for col in ['대출월', '대출년', 'RV신청일자', '최초카드론이용경과월', '최종카드론이용경과월']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

if '혜택수혜율_B0M' in df.columns:
    df['혜택수혜율_B0M'] = df['혜택수혜율_B0M'].fillna(df['혜택수혜율_B0M'].median())

for col in df.select_dtypes(include='float64').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='int64').columns:
    df[col] = df[col].fillna(0).astype(int)

# -------------------------------------------
# 🔸 8. 타입 최적화 및 저장
# -------------------------------------------
for col in df.select_dtypes(include='float64').columns:
    if df[col].dropna().apply(float.is_integer).all():
        df[col] = df[col].astype('Int64')

df.drop(columns=[col for col in df.columns if df[col].isna().all()], inplace=True)

df.to_csv(output_path, index=False)
print(f"✅ 전처리 완료: {output_path}")
print(f"📊 최종 shape: {df.shape}")