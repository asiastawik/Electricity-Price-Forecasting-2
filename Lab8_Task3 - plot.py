import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

data = pd.read_csv('GEFCOM.txt', delimiter="\s+", index_col=False, header=None, names=['YYYYMMDD', 'HH', 'zonal_price', 'system load', 'zonal_load', 'day-of-the-week'])
data_np = np.loadtxt('GEFCOM.txt', delimiter='\t', usecols=list(range(6)))
data['YYYYMMDD'] = data['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
#print(data)
indicies = []
for i in data.index:
    indicies.append(datetime.strftime(data.loc[i, 'YYYYMMDD'],'%d/%m %Y'))

train_data = data[data['YYYYMMDD'] < '2013-11-04'].copy()
test_data = data[data['YYYYMMDD'] >= '2013-11-04']
days_train = len(train_data)
days_test = len(test_data)
real_train = data_np[:days_train, 2]
real_test = data_np[-days_test:, 2]

#ARX(7) rolling
ar1 = np.zeros((int(days_test/24), 24))
days_train_ones = int(days_train / 24 - 7)
days_test_ones = int(days_test/24)
real_test = test_data['zonal_price'].values

for hour in range(24):
    #print(train_data)
    train_data_hour = train_data[train_data['HH'] == hour]
    for i in range(days_test_ones):
        # y - labels for training (dependent variable)
        # x - inputs for training (independent variables)
        # xf - inputs for the test
        test_datum = test_data.iloc[i]
        y = train_data_hour['zonal_price'].values
        z = train_data_hour['zonal_load'].values
        #print(y)
        #print(train_data['zonal_price'])
        x1 = y[:-1].reshape(-1, 1)  # y shifted by 1 day back
        #print(x1)
        x2 = y[:-7].reshape(-1, 1)  # y shifted by 7 days back
        z = z[:].reshape(-1, 1)
        xf = np.array([y[-1], y[-7], z[0], 1], dtype=object).reshape(-1, 1)  # inputs for the test
        y = y[7:].reshape(-1, 1)  # remove first 7 days from y
        x = np.hstack([x1[6:], x2[:], z[7:], np.ones((len(y), 1))])  # create x by stacking x1, x2, and a column of ones
        #print(x)
        betas = np.linalg.lstsq(x, y, rcond=None)[0]  # estimate betas
        pred = np.dot(xf.T, betas)  # make prediction
        ar1[i, hour] = pred
        train_data = pd.concat([train_data, test_datum])

ar1 = np.reshape(ar1, (ar1.shape[0] * ar1.shape[1],))
#print(ar1)
#print(len(ar1))
print(['ARX(7) rolling MAE', np.mean(np.abs(ar1 - real_test))])
print(['ARX(7) rolling RMSE', np.sqrt(np.mean((ar1 - real_test)**2))])


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
ax2.plot(ar1)
ax2.set_xticks(ticks=[1, 169, 337, 505, 673, 841, 1009])
ax2.set_xticklabels(labels=['04_11_2013', '11_11_2013', '18_11_2013', '25_11_2013', '02_12_2013', '09_12_2013', '16_12_2013'])
ax2.set_title('ARX(7) rolling forecast')

plt.show()
