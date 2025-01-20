import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

#데이터 로드
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
sample_submission = pd.read_csv('./dataset/sample_submission.csv')

#결측값 처리
#수치형 변수: 평균 또는 중앙값으로 대체
train['전용면적'] = train['전용면적'].fillna(train['전용면적'].mean())
train['해당층'] = train['해당층'].fillna(train['해당층'].median())
train['총층'] = train['총층'].fillna(train['총층'].median())
train['방수'] = train['방수'].fillna(train['방수'].median())
train['욕실수'] = train['욕실수'].fillna(train['욕실수'].median())
train['총주차대수'] = train['총주차대수'].fillna(0)  #결측값 0으로 대체

#범주형 변수: 최빈값으로 대체
train['주차가능여부'] = train['주차가능여부'].fillna(train['주차가능여부'].mode()[0])

#테스트 데이터도 동일한 방식으로 결측값 처리
test['전용면적'] = test['전용면적'].fillna(train['전용면적'].mean())
test['해당층'] = test['해당층'].fillna(train['해당층'].median())
test['총층'] = test['총층'].fillna(train['총층'].median())
test['방수'] = test['방수'].fillna(train['방수'].median())
test['욕실수'] = test['욕실수'].fillna(train['욕실수'].median())
test['총주차대수'] = test['총주차대수'].fillna(0)
test['주차가능여부'] = test['주차가능여부'].fillna(train['주차가능여부'].mode()[0])

#날짜 변환 및 파생 변수 생성
for df in [train, test]:
    df['게재일'] = pd.to_datetime(df['게재일'])
    df['게재년도'] = df['게재일'].dt.year
    df['게재월'] = df['게재일'].dt.month
    df['게재일차'] = df['게재일'].dt.day

#범주형 변수 인코딩
#Train과 Test 데이터를 합친 뒤 Label Encoding 수행
combined_data = pd.concat([train, test], axis=0)

#범주형 변수 처리
for col in ['매물확인방식', '방향', '주차가능여부', '중개사무소', '제공플랫폼']:
    label_encoder = LabelEncoder()
    combined_data[col] = label_encoder.fit_transform(combined_data[col])

#Train/Test 데이터 분리
train = combined_data.iloc[:len(train), :].reset_index(drop=True)
test = combined_data.iloc[len(train):, :].reset_index(drop=True)

#불필요한 열 제거
X = train.drop(['ID', '허위매물여부', '게재일'], axis=1)
y = train['허위매물여부']

#데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#모델 학습
model = RandomForestClassifier(random_state=42, class_weight='balanced')  #클래스 불균형 고려
model.fit(X_train, y_train)

#모델 예측 및 평가
y_pred = model.predict(X_val)
print('Macro F1 Score:', f1_score(y_val, y_pred, average='macro'))

#테스트 데이터 예측 및 제출 파일 생성
X_test = test.drop(['ID', '허위매물여부', '게재일'], axis=1)
test_predictions = model.predict(X_test)
sample_submission['허위매물여부'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
print("제출 파일 생성 완료: submission.csv")

#예측 결과 분포 확인
print(sample_submission['허위매물여부'].value_counts())
