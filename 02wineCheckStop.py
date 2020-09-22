# wineCheckStop.py
import pandas as pd

'''
콜백 함수 : 시스템이 어떠한 요구 사항을 충족했을 때, 스스로 동작하는 함수
ModelCheckPoint : 매 epoch마다 모델을 자동으로 저장해주는 콜백 함수
EarlyStopping : 학습이 진행되는 동안 개선의 여지가 보이지 않으면 강제로 종료
'''

'''
파일을 읽어 들여서 전체 15%만 샘플링한다.
모델을 생성한 다음 학습시킨다.
매 epoch마다 모델을 학습시켜 업데이트 한 후 해당 모델을 저장한다.
콜백 함수에 대한 객체를 생성한다.
fit()함수의 callbacks 매개 변수에 콜백 객체를 대입해준다.
History 객체를 이용하여 비용 함수와 정확도 정보를 확인한다.
비용 함수와 정확도에 대한 시각화를 수행한다.
'''

filename = '02wine.csv'
df_wine = pd.read_csv(filename, header=None)

print(df_wine.shape)
print('-' * 40)

# sample: 샘플링을 수행해주는 함수
# frac : fraction
df = df_wine.sample(frac=0.15)
print(df.shape)

data = df.values

table_col = data.shape[1]

y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column]

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=8, activation='relu'))
# tanh
model.add(Dense(units=y_column, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import os
model_dir = './model/' # 모델을 저장할 폴더

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# epoch: 에포크
# val_loss: 검증용 데이터 셋의 손실오차
model_name = model_dir + '{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping

# filepath: 파일을 저장할 경로, monitor: 모니터링할 대상
# loss(학습 셋 오차), val_loss(검증용 셋 오차), acc(학습 정확도), val_acc(검증용 셋 정확도)
# save_best_only: 이전 단계보다 모델이 더 개선되었을 경우에만 저장
mcp = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)

# patience=100: 테스트의 오차가 개선되지 않더라도 100번 기다리겠습니다.
es = EarlyStopping(monitor='val_loss', patience=100)

# validation_split: 훈련에 반영하지 않고, 테스트 용으로 사용할 데이터 셋의 비율
#                   실제 학습에 사용되지는 않고, 매 epoch의 끝 자락에서 metrics를 평가하는 용도로 사용

# callbakcs: 콜백 함수 객체들을 리스트 형식으로 지정해 준다.
history = model.fit(x, y, validation_split=0.2, epochs=3500, batch_size=500, callbacks=[es, mcp])

# History: fit()함수의 반환 객체
val_loss = history.history['val_loss']
print(val_loss)
print('-' * 30)

accuracy = history.history['accuracy']
print(accuracy)
print('-' * 30)

import numpy as np
import matplotlib.pyplot as plt

x_len = np.arange(len(accuracy))

plt.plot(x_len, val_loss, 'o', c='red', markersize=3)
plt.plot(x_len, accuracy, 'o', c='blue', markersize=3)

filename = '02wineTest.png'
plt.savefig(filename)
print(filename + '파일 저장됨')