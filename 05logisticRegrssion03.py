# logisticRegression03.py
import pandas as pd

filename = '05pima-indians-diabetes.csv'
data = pd.read_csv(filename)
print(data.shape)

print(data.columns)

print(data['class'].unique())

inputData = ['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age']
x_data = data[inputData]
y_data = data['class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# 모델 생성하기
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
print('# train 정확도 : %.3f' % (train_score))

test_score = model.score(x_test, y_test)
print('# test 정확도 : %.3f' % (test_score))

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ')
print(type(model.coef_))
coef = model.coef_.ravel()

import numpy as np
maxidx = np.argmax(coef.tolist())
print('가중치가 가장 큰 컬럼 : ', inputData[maxidx])

print(coef.tolist())

mylist = [] # 컬럼 이름과 가중치 정보를 담고 있는 리스트
for idx in range(len(coef.tolist())):
    mylist.append((inputData[idx], coef.tolist()[idx]))
print(mylist)

print('절편 : ')
print(model.intercept_)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print('test results:')

prediction = model.predict(x_test)
print('confusion matrix:')

cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
print('-' * 40)

accuracy = accuracy_score(y_test, prediction)
print('\n정확도 : %.3f' % (100 * accuracy))
print('\nclassification report:')
cl_report = classification_report(y_test, prediction)
print(cl_report)
print('-' * 40)

# 히트맨 생성
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

# plt.figure()
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap='YlGnBu', fmt='g')

# plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predict label")
# plt.show()

filename = '05logisticRegression03_01.png'
plt.savefig( filename, dpi=400, bbox_inches='tight' )
print( filename + ' 파일이 저장되었습니다.')

print('finished')