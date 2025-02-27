import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

# 데이터 로드
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
sample_submission = pd.read_csv('./dataset/sample_submission.csv')

# 결측값 처리
for col in ['전용면적', '해당층', '총층', '방수', '욕실수']:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

train['총주차대수'] = train['총주차대수'].fillna(0)
test['총주차대수'] = test['총주차대수'].fillna(0)

train['주차가능여부'] = train['주차가능여부'].fillna(train['주차가능여부'].mode()[0])
test['주차가능여부'] = test['주차가능여부'].fillna(train['주차가능여부'].mode()[0])

# 날짜 처리 및 특성 엔지니어링
for df in [train, test]:
    df['게재일'] = pd.to_datetime(df['게재일'])
    df['게재년도'] = df['게재일'].dt.year
    df['게재월'] = df['게재일'].dt.month
    df['게재일차'] = df['게재일'].dt.day
    df['면적당월세'] = df['월세'] / (df['전용면적'] + 1)
    df['층수비율'] = df['해당층'] / (df['총층'] + 1)

# One-Hot Encoding 및 데이터 준비
categorical_columns = ['매물확인방식', '방향', '주차가능여부', '중개사무소', '제공플랫폼']
train = pd.get_dummies(train, columns=categorical_columns)
test = pd.get_dummies(test, columns=categorical_columns)

# train과 test의 컬럼 일치시키기
all_columns = set(train.columns) | set(test.columns)
train = train.reindex(columns=all_columns, fill_value=0)
test = test.reindex(columns=all_columns, fill_value=0)

# 데이터 준비
X = train.drop(['ID', '허위매물여부', '게재일'], axis=1).astype('float32')
y = train['허위매물여부']
X_test = test.drop(['ID', '허위매물여부', '게재일'], axis=1).astype('float32')

# 하이퍼파라미터 탐색 공간 정의 (축소)
xgb_params = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'colsample_bytree': [0.6, 0.8]
}

lgb_params = {
    'num_leaves': [31, 63],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# 모델 정의 및 하이퍼파라미터 최적화 축소 적용
xgb_model = RandomizedSearchCV(xgb.XGBClassifier(random_state=42), xgb_params, n_iter=5, cv=3, random_state=42, n_jobs=-1)
lgb_model = RandomizedSearchCV(LGBMClassifier(random_state=42), lgb_params, n_iter=5, cv=3, random_state=42, n_jobs=-1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 교차 검증 설정 및 학습/예측
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_scores = []
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    val_preds_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
    
    test_preds_proba_fold_rf = rf_model.predict_proba(scaler.transform(X_test))[:, 1]
    test_predictions += test_preds_proba_fold_rf / n_splits
    
print(f"Mean ROC AUC: {np.mean(cv_scores):.4f}")

# 최종 예측 및 제출 파일 생성
final_predictions_test_set_rf=(test_predictions > 0.5).astype(int)

sample_submission['허위매물여부'] = final_predictions_test_set_rf
sample_submission.to_csv('submission_memory_optimized.csv', index=False)
print("제출 파일 생성 완료: submission_memory_optimized.csv")
