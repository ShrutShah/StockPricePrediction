import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

start = '2013-01-01'
end = '2023-12-01'
df = yf.download('WMT', start=start, end=end)

df = df.reset_index()

df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

plt.plot(df.Close)

ma100 = df.Close.rolling(100).mean()

plt.figure(figsize= (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.legend()

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): ])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler (feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

x_train = []
Y_train = []
for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100: i])
  Y_train.append(data_training_array[i, 0])
x_train, Y_train = np.array(x_train), np.array(Y_train)

from keras. layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout (0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout (0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout (0.4))

model.add(LSTM(units = 120, activation = 'relu', ))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,Y_train,epochs=50)

model.save('lstm_model.h5')

from keras.models import load_model
model = load_model('lstm_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]) :
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print (y_test.shape)

y_predicted = model. predict (x_test)
y_predicted.shape

scaler.scale_

scale_factor = 1/scaler.scale_
Y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

y_test

Y_predicted

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(Y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
