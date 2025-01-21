import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

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

# 불균형 데이터 처리 (SMOTE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 모델 학습
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 검증 데이터 예측
y_pred = model.predict(X_val)
print("Macro F1 Score:", f1_score(y_val, y_pred, average='macro'))
print(classification_report(y_val, y_pred))

# 테스트 데이터 예측 및 제출 파일 생성
test_predictions = model.predict(X_test)
sample_submission['허위매물여부'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
print("제출 파일 생성 완료: submission.csv")
