import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

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

# XGBoost, RandomForest, LightGBM 모델 설정
xg_model = xgb.XGBClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
lgb_model = LGBMClassifier(random_state=42)

# 각 모델 학습
xg_model.fit(X_train_resampled, y_train_resampled)
rf_model.fit(X_train_resampled, y_train_resampled)
lgb_model.fit(X_train_resampled, y_train_resampled)

# 각 모델 예측
y_pred_xg = xg_model.predict(X_val)
y_pred_rf = rf_model.predict(X_val)
y_pred_lgb = lgb_model.predict(X_val)

# 예측값 앙상블 (단순 평균)
y_pred_ensemble = np.round((y_pred_xg + y_pred_rf + y_pred_lgb) / 3)

# 성능 평가
print("Macro F1 Score (Ensemble):", f1_score(y_val, y_pred_ensemble, average='macro'))
print(classification_report(y_val, y_pred_ensemble))

# 최종 모델로 테스트 데이터 예측
test_predictions_ensemble = np.round((xg_model.predict(X_test) + rf_model.predict(X_test) + lgb_model.predict(X_test)) / 3)

# 제출 파일 생성
sample_submission['허위매물여부'] = test_predictions_ensemble
sample_submission.to_csv('submission_ensemble_optimized.csv', index=False)
print("제출 파일 생성 완료: submission_ensemble_optimized.csv")
