import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

data = pd.read_csv('GEFCOM.txt', delimiter="\s+", index_col=False, header=None, names=['YYYYMMDD', 'HH', 'zonal_price', 'system load', 'zonal_load', 'day-of-the-week'])
data_np = np.loadtxt('GEFCOM.txt', delimiter='\t', usecols=list(range(6)))
data['YYYYMMDD'] = data['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
#print(data)

#only weeks which started by Monday
data = data[data['YYYYMMDD'].dt.weekday == 0]

indicies = []
for i in data.index:
    indicies.append(datetime.strftime(data.loc[i, 'YYYYMMDD'],'%d/%m %Y'))

train_data = data[data['YYYYMMDD'] < '2013-01-01']
#print(train_data)
test_data = data[data['YYYYMMDD'] >= '2013-01-01']
days_train = len(train_data)
days_test = len(test_data)
real_train = data_np[:days_train, 2]
real_test = data_np[-days_test:, 2]

#AR(7) single
real_test = test_data['zonal_price'].to_numpy() # true labels for test data
ar3 = np.zeros_like(real_test) # predefine output matrix
Ndays = len(ar3) // 24 #divide
firstind = 24 # the first index for our y_train (we omit the first day, as we can't create a training sample
firstind_7 = 24*7
lastind = np.argwhere(data_np[:, 0] == 20130101).squeeze()[0] # we need to select the first one
for day in range(Ndays):
    y = data_np[firstind_7 + 24*day:lastind + 24*day, 2] # zonal price is column with index 2
    x1 = data_np[firstind_7 + 24*(day-1):lastind + 24*day, 2] # y shifted by 1 day back, longer by 1 day (we will use the added samples for xf)
    x2 = data_np[firstind_7 + 24*(day-7):lastind + 24*day-6*24, 2] # y shifted by 7 days back, shorter by 6 days (we will use the added samples for xf)
    #print(len(x1), len(x2))
    x = np.stack([x1, x2, np.ones_like(x1)]).T # intercept added, .T means transpose to matrix
    xf = x[-24:, :] # extracting the xf, selects the last 24 samples of x, which will be used as the input to predict the next day's prices
    x = x[:-24, :] # remove xf from x, removes the last 24 samples of x, which will be used to train the model
    #data_np = np.column_stack((data_np, np.zeros(data_np.shape[0])))
    for h in range(24): #loop iterates over each hour in the day
        betas = np.linalg.lstsq(x[h::24], y[h::24], rcond=None)[0] # hourly data selected here
        '''x[h::24] selects every 24th element of x, starting from index h. 
        This creates a new array containing every h-th hourly zonal price in x,
         starting from the h-th hour of the day.'''
        ar3[day*24 + h] = np.dot(xf[h::24], betas)

print(['AR(7) single MAE', np.mean(np.abs(ar3 - real_test))])
print(['AR(7) single RMSE', np.sqrt(np.mean((ar3 - real_test)**2))])

# calculate MAE and RMSE for each week
mae_weeks = []
rmse_weeks = []
for i in range(Ndays // 7):
    start = i * 7 * 24
    end = start + 7 * 24
    mae_weeks.append(np.mean(np.abs(ar3[start:end] - real_test[start:end])))
    rmse_weeks.append(np.sqrt(np.mean((ar3[start:end] - real_test[start:end])**2)))

# plot MAE and RMSE for each week
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(mae_weeks))
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, mae_weeks, bar_width,
                alpha=opacity, color='b',
                label='MAE for AR(7) single')

rects2 = ax.bar(index + bar_width, rmse_weeks, bar_width,
                alpha=opacity, color='g',
                label='RMSE for AR(7) single')

ax.set_xlabel('Week')
ax.set_ylabel('Error')
ax.set_title('AR(7) single: MAE and RMSE for each week')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(np.arange(1, len(mae_weeks)+1))
ax.legend()

fig.tight_layout()
plt.show()

# find weeks with lowest and highest scores for MAE
mae_lowest = np.argmin(mae_weeks)
mae_highest = np.argmax(mae_weeks)
print("MAE lowest week: {}, Error: {}".format(mae_lowest+1, mae_weeks[mae_lowest]))
print("MAE highest week: {}, Error: {}".format(mae_highest+1, mae_weeks[mae_highest]))

# find weeks with lowest and highest scores for RMSE
rmse_lowest = np.argmin(rmse_weeks)
rmse_highest = np.argmax(rmse_weeks)
print("RMSE lowest week: {}, Error: {}".format(rmse_lowest+1, rmse_weeks[rmse_lowest]))
print("RMSE highest week: {}, Error: {}".format(rmse_highest+1, rmse_weeks[rmse_highest]))


#AR(7) rolling
days_train_ones = int(days_train / 24 - 7)
days_test_ones = int(days_test/24)
real_test = test_data['zonal_price'].values
ar2 = np.zeros((int(days_test/24), 24))

for hour in range(24):
    #print(train_data)
    train_data_hour = train_data[train_data['HH'] == hour]
    for i in range(days_test_ones):
        # y - labels for training (dependent variable)
        # x - inputs for training (independent variables)
        # xf - inputs for the test
        test_datum = test_data.iloc[i]
        y = train_data_hour['zonal_price'].values
        #print(y)
        #print(train_data['zonal_price'])
        x1 = y[:-1].reshape(-1, 1)  # y shifted by 1 day back
        #print(x1)
        x2 = y[:-7].reshape(-1, 1)  # y shifted by 7 days back
        xf = np.array([y[-1], y[-7], 1]).reshape(-1, 1)  # inputs for the test
        y = y[7:].reshape(-1, 1)  # remove first 7 days from y
        x = np.hstack([x1[6:], x2[:], np.ones((len(y), 1))])  # create x by stacking x1, x2, and a column of ones
        #print(x)
        betas = np.linalg.lstsq(x, y, rcond=None)[0]  # estimate betas
        pred = np.dot(xf.T, betas)  # make prediction
        ar2[i, hour] = pred
        train_data = pd.concat([train_data, test_datum])

ar2 = np.reshape(ar2, (ar2.shape[0] * ar2.shape[1],))
print(['AR(7) rolling MAE', np.mean(np.abs(ar2 - real_test))])
print(['AR(7) rolling RMSE', np.sqrt(np.mean((ar2 - real_test)**2))])

# calculate MAE and RMSE for each week
mae_weeks = []
rmse_weeks = []
for i in range(Ndays // 7):
    start = i * 7 * 24
    end = start + 7 * 24
    mae_weeks.append(np.mean(np.abs(ar2[start:end] - real_test[start:end])))
    rmse_weeks.append(np.sqrt(np.mean((ar2[start:end] - real_test[start:end])**2)))

# plot MAE and RMSE for each week
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(mae_weeks))
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, mae_weeks, bar_width,
                alpha=opacity, color='b',
                label='MAE for AR(7) rolling')

rects2 = ax.bar(index + bar_width, rmse_weeks, bar_width,
                alpha=opacity, color='g',
                label='RMSE for AR(7) rolling')

ax.set_xlabel('Week')
ax.set_ylabel('Error')
ax.set_title('AR(7) rolling: MAE and RMSE for each week')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(np.arange(1, len(mae_weeks)+1))
ax.legend()

fig.tight_layout()
plt.show()

# find weeks with lowest and highest scores for MAE
mae_lowest = np.argmin(mae_weeks)
mae_highest = np.argmax(mae_weeks)
print("MAE lowest week: {}, Error: {}".format(mae_lowest+1, mae_weeks[mae_lowest]))
print("MAE highest week: {}, Error: {}".format(mae_highest+1, mae_weeks[mae_highest]))

# find weeks with lowest and highest scores for RMSE
rmse_lowest = np.argmin(rmse_weeks)
rmse_highest = np.argmax(rmse_weeks)
print("RMSE lowest week: {}, Error: {}".format(rmse_lowest+1, rmse_weeks[rmse_lowest]))
print("RMSE highest week: {}, Error: {}".format(rmse_highest+1, rmse_weeks[rmse_highest]))


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
print(['ARX(7) rolling MAE', np.mean(np.abs(ar1 - real_test))])
print(['ARX(7) rolling RMSE', np.sqrt(np.mean((ar1 - real_test)**2))])

# calculate MAE and RMSE for each week
mae_weeks = []
rmse_weeks = []
for i in range(Ndays // 7):
    start = i * 7 * 24
    end = start + 7 * 24
    mae_weeks.append(np.mean(np.abs(ar1[start:end] - real_test[start:end])))
    rmse_weeks.append(np.sqrt(np.mean((ar1[start:end] - real_test[start:end])**2)))

# plot MAE and RMSE for each week
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(mae_weeks))
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, mae_weeks, bar_width,
                alpha=opacity, color='b',
                label='MAE for ARX(7) rolling')

rects2 = ax.bar(index + bar_width, rmse_weeks, bar_width,
                alpha=opacity, color='g',
                label='RMSE for ARX(7) rolling')

ax.set_xlabel('Week')
ax.set_ylabel('Error')
ax.set_title('ARX(7) rolling: MAE and RMSE for each week')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(np.arange(1, len(mae_weeks)+1))
ax.legend()

fig.tight_layout()
plt.show()

# find weeks with lowest and highest scores for MAE
mae_lowest = np.argmin(mae_weeks)
mae_highest = np.argmax(mae_weeks)
print("MAE lowest week: {}, Error: {}".format(mae_lowest+1, mae_weeks[mae_lowest]))
print("MAE highest week: {}, Error: {}".format(mae_highest+1, mae_weeks[mae_highest]))

# find weeks with lowest and highest scores for RMSE
rmse_lowest = np.argmin(rmse_weeks)
rmse_highest = np.argmax(rmse_weeks)
print("RMSE lowest week: {}, Error: {}".format(rmse_lowest+1, rmse_weeks[rmse_lowest]))
print("RMSE highest week: {}, Error: {}".format(rmse_highest+1, rmse_weeks[rmse_highest]))
