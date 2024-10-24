import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

data = pd.read_csv('GEFCOM.txt', delimiter="\s+", index_col=False, header=None, names=['YYYYMMDD', 'HH', 'zonal_price', 'system_load', 'zonal_load', 'day-of-the-week'])
data_np = np.loadtxt('GEFCOM.txt', delimiter='\t', usecols=list(range(6)))
data['YYYYMMDD'] = data['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
#print(data)
indicies = []
for i in data.index:
    indicies.append(datetime.strftime(data.loc[i, 'YYYYMMDD'],'%d/%m %Y'))

train_data = data[data['YYYYMMDD'] < '2013-11-04'].copy()
test_data = data[data['YYYYMMDD'] >= '2013-11-04']
days_train = int(len(train_data) / 24)
days_test = int(len(test_data) / 24)
real_train = data_np[:days_train, 2]
real_test = data_np[-days_test:, 2]

train_data_np = train_data['zonal_price'].to_numpy()
test_data_np = test_data['zonal_price'].to_numpy()

#ARX(7) rolling
ar3 = []

start = 0
for day in range(days_test):
    train = data.iloc[start:17544 + start]
    train_data_zonal = train['zonal_price']
    train_data_system = train['system_load']

    for hour in range(24):
        y = train_data_zonal.iloc[hour::24]
        z = train_data_system.iloc[hour::24]
        x = np.stack([np.ones((len(train_data_zonal)) // 24 - 7), y.iloc[6:-1], y.iloc[:-7], z.iloc[7:]])
        temp = data['zonal_price']
        temp_z = data['system_load']
        a = (len(train_data_zonal) + start) // 24
        xf = np.stack([np.ones((1,)), temp.iloc[hour::24].iloc[a - 1:a], temp.iloc[hour::24].iloc[a - 7:a - 6],
                       temp_z.iloc[hour::24].iloc[a:a + 1]])
        y = y.iloc[7:]
        betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
        prediction = np.dot(betas, xf)
        pred = prediction.reshape(1, len(prediction))
        ar3.append(pred[0, 0])
    start += 24

print(['ARX(7) rolling MAE', np.mean(np.abs(ar3 - test_data_np))])
print(['ARX(7) rolling RMSE', np.sqrt(np.mean((ar3 - test_data_np)**2))])

#TASK 3
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
#fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(data.zonal_price)
ax1.set_xticks(ticks=[24913, 25057, 25249, 25417, 25585, 25753, 25921])
ax1.set_xticklabels(labels=['04_11_2013', '11_11_2013', '18_11_2013', '25_11_2013', '02_12_2013', '09_12_2013', '16_12_2013'])
ax1.set_xlim(24913, len(data))
#axs[0].tick_params(axis='x', labelbottom=True)
ax1.set_title('Zonal prices')

#fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(ar3)
ax2.set_xticks(ticks=[72, 240, 408, 576, 744, 912, 1080])
ax2.set_xticklabels(labels=['04_11_2013', '11_11_2013', '18_11_2013', '25_11_2013', '02_12_2013', '09_12_2013', '16_12_2013'])
ax2.set_title('ARX(7) rolling forecast')

plt.show()
