# logisticRegression02.py
import pandas as pd

filename = '04titanic.csv'
data = pd.read_csv(filename)

# Survived 컬럼: 생사 여부(1이 생존)
print(data.columns)
print('-' * 30)

# 코딩 변경) 데이터 분석에 맞도록 사용자가 값을 변경하는 작업 (문자를 숫자로)
print(data['Sex'].unique()) # 중복된 데이터는 배제하고 1개만 표시

data['Sex'] = data['Sex'].map({'female':1, 'male':0})
print(data['Sex'].unique())

# 결측치(missing value): 누락된 데이터
# 출력 결과에 nan이라고 표현되는 부분을 말한다.
print(data['Age'].unique())
# 결측치를 나머지 항목들의 평균값으로 치환한다.
# fillna: 결측치(na)를 채워 넣기(fill)
data['Age'].fillna(value=data['Age'].mean(), inplace=True)
print(data['Age'].unique())

print(data['Survived'].unique())

print(data['Pclass'].unique()) # 좌석 등급

result = data.groupby('Pclass')['Pclass'].count()
print('좌석 등급별 인원 수')
print(result)

# dummy 컬럼 만들기
# 파생 컬럼 생성: FirstClass(1등석일때 1), SecondClass(2등석일때 1)

# FirstClass 컬럼에는 Pclass의 값이 1일때만 1, 나머지는 0
data['FirstClass'] = data['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# SecondClass 컬럼에는 Pclass의 값이 1일때만 1, 나머지는 0
data['SecondClass'] = data['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# 성별과 나이, 일등성/이등석 정보를 이용하여 생존 여부 판단하기
x_data = data[['Sex', 'Age', 'FirstClass', 'SecondClass']]
y_data = data['Survived']

# 데이터 셋 분리 작업
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# fit_transform 함수를 사용하여 이미 한번 fit했으므로 두 번째 문장에서는
# transform 함수만 호출하면 된다.
x_test = scaler.transform(x_test)

# 모델 생성하기
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# 주의: 상관 분석은 인과 관계가 아니라 연관성과 관련 있다.

# coef_는 w값을 의미한다.
# 회귀분석에서 w의 의미는 y에 대한 인과 관계의 정도를 의미한다.
# 성별과 일등석 탑승 여부가 생사를 판단하는데 가장 영향력을 많이 준다고 볼 수 있다.
print(model.coef_)
print('-' * 30)

print(model.intercept_)
print('-' * 30)

# 샘플 데이터로 예측하기
import numpy as np

# 성별, 나이, 1등석, 2등석
soo = np.array([0.0, 20.0, 0.0, 0.0])
hee = np.array([1.0, 17.0, 1.0, 0.0])
minho = np.array([0.0, 32.0, 1.0, 0.0])
sample = np.array([soo, hee, minho])

print(sample)

sample = scaler.transform(sample)

print(sample)

print(model.predict(sample))

# probability: 확률
proba = print(model.predict_proba(sample))

from sklearn.metrics import confusion_matrix

print('test result')
prediction = model.predict((x_test))

# confusion_matrix(정답데이터, 예측값)
cf_matrix = confusion_matrix(y_test, prediction)
print('confusion matrix : ')
print(cf_matrix)
print('-' * 30)

# accuracy_score: 정확도를 구해주는 함수
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print('accuracy(정확도) : %.3f' % (100 * accuracy))
print('-' * 30)

# 주요 분류와 관련된 metrics(지표)에 대한 보고서
from sklearn.metrics import classification_report

print('classification_report : ')
cl_report = classification_report(y_test, prediction)
print(cl_report)
print('-' * 30)

# heatmap 도표
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap='YlGnBu', fmt='g')

plt.title('confusion matrix')
plt.ylabel('actual label')
plt.xlabel('prediction label')

filename = '04logisticRegression02_01.png'
plt.savefig(filename)
print(filename + '파일 저장됨')

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
