import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb  # XGBoost 추가
from scipy.stats import randint

# 데이터 로드
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
sample_submission = pd.read_csv('./dataset/sample_submission.csv')

# 결측값 처리
for col in ['전용면적', '해당층', '총층', '방수', '욕실수']:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(train[col].mean())

train['총주차대수'] = train['총주차대수'].fillna(0)
test['총주차대수'] = test['총주차대수'].fillna(0)

train['주차가능여부'] = train['주차가능여부'].fillna(train['주차가능여부'].mode()[0])
test['주차가능여부'] = test['주차가능여부'].fillna(train['주차가능여부'].mode()[0])

# 날짜 처리
for df in [train, test]:
    df['게재일'] = pd.to_datetime(df['게재일'])
    df['게재년도'] = df['게재일'].dt.year
    df['게재월'] = df['게재일'].dt.month
    df['게재일차'] = df['게재일'].dt.day

# 파생 변수 추가
for df in [train, test]:
    df['면적당월세'] = df['월세'] / df['전용면적']
    df['층수비율'] = df['해당층'] / df['총층']

# Train/Test 데이터 합치기
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis=0)

# One-Hot Encoding
combined = pd.get_dummies(combined, columns=['매물확인방식', '방향', '주차가능여부', '중개사무소', '제공플랫폼'])

# Train/Test 다시 분리
train = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
test = combined[combined['is_train'] == 0].drop(['is_train', '허위매물여부'], axis=1)

# 불필요한 열 제거
X = train.drop(['ID', '허위매물여부', '게재일'], axis=1)
y = train['허위매물여부']
X_test = test.drop(['ID', '게재일'], axis=1)

# 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE 불균형 데이터 처리 (과도한 샘플링 방지)
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# XGBoost 모델 설정
xg_model = xgb.XGBClassifier(random_state=42)

# 하이퍼파라미터 설정
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [3, 5, 7, 10, None],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 2, 5]  # 클래스 불균형 처리
}

# XGBoost 모델로 하이퍼파라미터 튜닝
xg_model = xgb.XGBClassifier(random_state=42)

# Cross-validation을 사용하여 하이퍼파라미터 튜닝 (XGBoost의 cv 사용)
cv_results = xgb.cv(
    params={'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    dtrain=xgb.DMatrix(X_train_resampled, label=y_train_resampled),
    num_boost_round=1000,
    nfold=5,
    metrics="error",
    early_stopping_rounds=50,
    as_pandas=True
)

# 최적 부스팅 라운드 찾기
best_num_boost_round = cv_results.shape[0] - 1  # 가장 마지막 부스팅 라운드

# XGBoost 모델 학습
best_model = xgb.XGBClassifier(
    n_estimators=best_num_boost_round,
    random_state=42,
    scale_pos_weight=2  # 클래스 불균형에 대한 가중치
)

best_model.fit(X_train_resampled, y_train_resampled)

# 최적 모델로 예측
y_pred = best_model.predict(X_val)
print("Macro F1 Score (XGBoost):", f1_score(y_val, y_pred, average='macro'))
print(classification_report(y_val, y_pred))

# XGBoost로 예측한 결과로 제출 파일 생성
test_predictions_xg = best_model.predict(X_test)
sample_submission['허위매물여부'] = test_predictions_xg
sample_submission.to_csv('submission_xgboost_optimized.csv', index=False)  # 최적화된 XGBoost 모델로 생성된 제출 파일
print("제출 파일 생성 완료: submission_xgboost_optimized.csv")
