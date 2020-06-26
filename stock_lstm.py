import numpy as np
import pandas as pd

samsung = np.load('C:/Users/JPark/Documents/Python Scripts/Stock/samsung.npy', allow_pickle=True)
date = pd.read_csv('C:/Users/JPark/Documents/Python Scripts/Stock/date.csv')

print(samsung)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            #print(x)
            #print(y)
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 0]    # 수정
        
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1) #time_steps는 RNN의 step 수, y_column은 x값에 대하여 y값 출력 위치
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)

# 데이터 셋 나누기
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3, shuffle = False)
date_train, date_test = train_test_split(
    date, random_state=1, test_size = 0.3, shuffle = False)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# StandardScalar가 (x,y) 형태의 data만 적용 가능하여 그에 맞도록 수정
x_train = np.reshape(x_train,
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)
print(x_test.shape)

#### 데이터 전처리 #####
#StandardScalar : 평균0, 표준편차 1
#RobustScalar : median 0, IQR(Interquartile Range) 1
#MinMaxScalar : max1, min 0
#MaxAbsScalar : 0을 기준으로 가잔 큰 수를 1, -1 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

x_train_scaled = np.reshape(x_train_scaled,
    (x_train_scaled.shape[0], 5, 6))
x_test_scaled = np.reshape(x_test_scaled,
    (x_test_scaled.shape[0], 5, 6))
print(x_train_scaled.shape)
print(x_test_scaled.shape)
print(np.mean(x_train_scaled))
print(np.std(x_train_scaled))

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(5, 6)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
          batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

#원래는 pred도 다른 데이터 써야 할텐데 지금은 그냥 test 그대로
y_pred = model.predict(x_test_scaled)

for i in range(len(y_test)):
    print('종가 : ', y_test[i], '/ 예측가 : ', y_pred[i])

#plotly
import plotly.offline as offline 
import plotly.graph_objs as go 


trace1 = go.Scatter( x=date.date, y=np.concatenate(np.concatenate([y_train, y_test])), name='test') 
trace2 = go.Scatter( x=date_test.date, y=np.concatenate(y_pred), name='pred') 

data = [trace1, trace2] 

layout = dict( 
            title='{}의 종가(close) Time Series'.format('samsung')            
        ) 
fig = go.Figure(data=data, layout=layout) 
offline.iplot(fig)




