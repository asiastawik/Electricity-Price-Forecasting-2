import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('GEFCOM.txt', delimiter="\s+", index_col=False, header=None, names=['YYYYMMDD', 'HH', 'zonal_price', 'system_load', 'zonal_load', 'day-of-the-week'])
data_np = np.loadtxt('GEFCOM.txt', delimiter='\t', usecols=list(range(6)))
data['YYYYMMDD'] = data['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
#print(data)
indicies = []
for i in data.index:
    indicies.append(datetime.strftime(data.loc[i, 'YYYYMMDD'],'%d/%m %Y'))

train_data = data[data['YYYYMMDD'] < '2013-01-01'].copy()
#print(train_data)
test_data = data[data['YYYYMMDD'] >= '2013-01-01']

train_data_np = train_data['zonal_price'].to_numpy()
test_data_np = test_data['zonal_price'].to_numpy()


#AR(7) single
ar1 = np.zeros((len(test_data)//24, 24))
ar1_single = np.zeros((len(test_data), 1))
test_data_np_len = len(test_data_np)
days_24 = 17544//24

for hour in range(24):
    y = train_data_np[hour::24]
    x = np.stack([np.ones((724,)), y[6:-1], y[:-7]])
    data_zonal = data['zonal_price']
    xf = np.stack([np.ones((test_data_np_len//24,)), data_zonal[hour::24].iloc[days_24-1:-1], data_zonal[hour::24].iloc[days_24-7:-7]])
    y = y[7:]
    betas = np.linalg.lstsq(x.T, y, rcond=None)[0]
    '''x[h::24] selects every 24th element of x, starting from index h. 
    This creates a new array containing every h-th hourly zonal price in x,
    starting from the h-th hour of the day.'''
    prediction = np.dot(betas, xf)
    ar1[:, hour] = prediction
    pred_len = len(prediction)
    pred = prediction.reshape(1, pred_len)
    ar1_single[hour::24] = pred.T

print(['AR(7) single MAE', np.mean(np.abs(ar1_single[:,0] - test_data_np))])
print(['AR(7) single RMSE', np.sqrt(np.mean((ar1_single[:,0] - test_data_np)**2))])

# calculate MAE and RMSE for each week

full_weeks_ar1 = ar1_single[144:8376]

mae_weeks = []
rmse_weeks = []

for i in range(len(full_weeks_ar1) // 168):
    start = i * 7 * 24
    end = start + 7 * 24
    mae_weeks.append(np.mean(np.abs(ar1_single[start:end,0] - test_data_np[start:end])))
    rmse_weeks.append(np.sqrt(np.mean((ar1_single[start:end,0] - test_data_np[start:end])**2)))

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
real_test = test_data['zonal_price'].to_numpy() # true labels for test data
ar2 = np.zeros_like(real_test) # predefine output matrix
Ndays = len(ar2) // 24 #divide
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
        ar2[day*24 + h] = np.dot(xf[h::24], betas)

print(['AR(7) rolling MAE', np.mean(np.abs(ar2 - test_data_np))])
print(['AR(7) rolling RMSE', np.sqrt(np.mean((ar2 - test_data_np)**2))])

# calculate MAE and RMSE for each week
full_weeks_ar2 = ar2[144:8376]
mae_weeks = []
rmse_weeks = []

for i in range(len(full_weeks_ar2) // 168):
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
days_train = int(len(train_data) / 24)
days_test = int(len(test_data) / 24)
data_zonal = data['zonal_price']
data_system = data['system_load']
ar3 = []
start = 0
for day in range(days_test):
    train = data.iloc[start:days_24*24 + start]
    train_data_zonal = train['zonal_price']
    train_data_system = train['system_load']
    data_len = len(train_data_zonal) // 24 - 7
    data_len_start = (len(train_data_zonal) + start) // 24
    for hour in range(24):
        y = train_data_zonal.iloc[hour::24]
        z = train_data_system.iloc[hour::24]
        x = np.stack([np.ones(data_len), y.iloc[6:-1], y.iloc[:-7], z.iloc[7:]])
        xf = np.stack([np.ones((1,)), data_zonal.iloc[hour::24].iloc[data_len_start - 1:data_len_start], data_zonal.iloc[hour::24].iloc[data_len_start - 7:data_len_start - 6],
                       data_system.iloc[hour::24].iloc[data_len_start:data_len_start + 1]])
        y = y.iloc[7:]
        betas = np.linalg.lstsq(x.T, y, rcond=None)[0]  # estimate betas
        prediction = np.dot(betas, xf)
        pred_reshaped = prediction.reshape(1, len(prediction))
        ar3.append(pred_reshaped[0, 0])
    start += 24 # increasing by one day (24 hours)

print(['ARX(7) rolling MAE', np.mean(np.abs(ar3 - test_data_np))])
print(['ARX(7) rolling RMSE', np.sqrt(np.mean((ar3 - test_data_np)**2))])

# calculate MAE and RMSE for each week
full_weeks_ar3 = ar3[144:8376]

mae_weeks = []
rmse_weeks = []
for i in range(len(full_weeks_ar3) // 168):
    start = i * 7 * 24
    end = start + 7 * 24
    mae_weeks.append(np.mean(np.abs(ar3[start:end] - test_data_np[start:end])))
    rmse_weeks.append(np.sqrt(np.mean((ar3[start:end] - test_data_np[start:end])**2)))

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

#TASK 3
fig, ax = plt.subplots(figsize=(12, 8))

data1 = (data.zonal_price[24840:]).to_numpy()
#print(data1)
ax.plot(data1, label='Zonal prices')
#ax.set_xticks(ticks=[24913, 25057, 25249, 25417, 25585, 25753, 25921])
ax.set_xticklabels(labels=['04_11_2013', '11_11_2013', '18_11_2013', '25_11_2013', '02_12_2013', '09_12_2013', '16_12_2013'])
#ax.set_xlim(24913, len(data))

data2 = ar3[7296:]
ax.plot(data2, label='ARX(7) rolling forecast')
ax.set_xticks(ticks=[72, 240, 408, 576, 744, 912, 1080])
ax.set_xticklabels(labels=['04_11_2013', '11_11_2013', '18_11_2013', '25_11_2013', '02_12_2013', '09_12_2013', '16_12_2013'])

ax.set_xlabel('Date')
ax.set_ylabel('Zonal price or its forecast')
ax.legend()
ax.set_title('Zonal prices and ARX(7) rolling forecast')

plt.show()
