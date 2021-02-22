import matplotlib
import pandas as pd
from django.shortcuts import render
import plotly.offline as pyoff
import plotly.graph_objs as go
import requests
import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

matplotlib.use('Agg')

pd.set_option('display.max_colwidth', 60)

# ARIMA MODEL FOR FAULT PREDICTION
import requests

url = "http://182.18.164.19:3402/Trans/CurrentVoltage/868997035786613/all"
user = "admin"
passwd = "admin@123"
auth_values = (user, passwd)
response = requests.get(url, auth=auth_values)
lis = response.json()
head = list(lis[0].keys())
head
rawdata = []
for i in range(0, len(lis)):
    l = []
    for x in head:
        if (x == 'DeviceImei' or x == 'A_id'):
            l.append(lis[i][x])
        elif (x == 'DeviceTimeStamp'):
            s = lis[i][x]
            l.append(s[0:16])
        else:
            l.append(float(lis[i][x]))
    rawdata.append(l)
    # rint(lis[0][x], type(lis[0][x]))


def Reverse(lst):
    return [ele for ele in reversed(lst)]


data = Reverse(rawdata)
cv = pd.DataFrame(data, columns=head)
cv['DeviceTimeStamp'] = pd.to_datetime(cv['DeviceTimeStamp'])
x = cv['DeviceTimeStamp'].max()
y = cv['DeviceTimeStamp'].min()
days = datetime.timedelta(7)
week = x - days
inutnxtweek = cv[cv['DeviceTimeStamp'] >= week]
inutnxtweek_high = inutnxtweek[inutnxtweek['INUT'] > 0.1 * 180]
total_inutnxtwk_high = len(inutnxtweek_high)
cv_inut_high = cv.query('INUT>0.1*180', inplace=False)
total_inut_high = len(cv_inut_high)
# ARIMA

dataset_inut = inutnxtweek['INUT'].values
dataset_inut = dataset_inut.astype('float32')

# fit model
model_inut = ARIMA(inutnxtweek['INUT'], order=(5, 1, 0))
model_fit_inut = model_inut.fit(disp=0)
X = inutnxtweek['INUT'].values
size = int(len(X) * 6 / 7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
confidence = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output, error, conf = model_fit.forecast()
    yhat = output[0]
    predictions.append([yhat, conf[0][0], conf[0][1]])
    obs = test[t]
    history.append(obs)
pred = pd.DataFrame(predictions)
error = mean_squared_error(test, pred[0])
rmse_inut = np.sqrt(mean_squared_error(test, pred[0]))
x_data_inut = pd.DataFrame(inutnxtweek['INUT'])
test1 = test.reshape(-1, 1)
test1 = pd.DataFrame(test1)
predictions1 = pd.DataFrame(pred[0])
low = pd.DataFrame(pred[1])
up = pd.DataFrame(pred[2])
train1 = pd.DataFrame(train)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(x_data_inut)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train), :] = train1

testPredictPlot = np.empty_like(x_data_inut)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train1):len(x_data_inut), :] = predictions1

lower = np.empty_like(x_data_inut)
lower[:, :] = np.nan
lower[len(train1):len(x_data_inut), :] = low

upper = np.empty_like(x_data_inut)
upper[:, :] = np.nan
upper[len(train1):len(x_data_inut), :] = up

testpp = pd.DataFrame(testPredictPlot)
trainpp = pd.DataFrame(trainPredictPlot)
df2 = pd.DataFrame(inutnxtweek['DeviceTimeStamp'])
df2 = df2.reset_index()
df2 = df2.drop(['index'], axis=1)
training = df2.merge(trainpp, left_index=True, right_index=True)
predicted = df2.merge(testpp, left_index=True, right_index=True)
original1_inut = inutnxtweek
original1_inut = original1_inut.set_index('DeviceTimeStamp')
trn_inut = training
trn_inut = trn_inut.set_index('DeviceTimeStamp')
prd_inut = predicted
prd_inut = prd_inut.set_index('DeviceTimeStamp')
fault_pred_inut = prd_inut[prd_inut[0] > 0.1 * 180]
inut_faults_next_day = len(fault_pred_inut)
s_inut = ''
if (inut_faults_next_day > 0):
    s_inut = 'Fault may occur next 24hrs'
else:
    s_inut = 'Fault may not occur next 24hrs'
upb = pd.DataFrame(upper)
ub_inut = df2.merge(upb, left_index=True, right_index=True)
ub_inut = ub_inut.set_index('DeviceTimeStamp')
lwb = pd.DataFrame(lower)
lb_inut = df2.merge(lwb, left_index=True, right_index=True)
lb_inut = lb_inut.set_index('DeviceTimeStamp')

# LOAD PREDICTION
x = cv['DeviceTimeStamp'].max()
y = cv['DeviceTimeStamp'].min()
days = datetime.timedelta(7)
week = x - days
il1nxtweek = cv[cv['DeviceTimeStamp'] >= week]
il1nxtweek_high = il1nxtweek.query('IL1>=1.2*180', inplace=False)
total_il1nxtwk_high = len(il1nxtweek_high)
cv_il1_high = cv.query('IL1>=1.2*180', inplace=False)
total_il1_high = len(cv_il1_high)
dataset_il1 = il1nxtweek['IL1'].values
dataset_il1 = dataset_il1.astype('float32')
# fit model
model_il1 = ARIMA(il1nxtweek['IL1'], order=(5, 1, 0))
model_fit_il1 = model_il1.fit(disp=0)
X = il1nxtweek['IL1'].values
size = int(len(X) * 6 / 7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
confidence = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output, error, conf = model_fit.forecast()
    yhat = output[0]
    predictions.append([yhat, conf[0][0], conf[0][1]])
    obs = test[t]
    history.append(obs)
pred = pd.DataFrame(predictions)
error = mean_squared_error(test, pred[0])
rmse_il1 = np.sqrt(mean_squared_error(test, pred[0]))
x_data_il1 = pd.DataFrame(il1nxtweek['IL1'])
test1 = test.reshape(-1, 1)
test1 = pd.DataFrame(test1)
predictions1 = pd.DataFrame(pred[0])
low = pd.DataFrame(pred[1])
up = pd.DataFrame(pred[2])
train1 = pd.DataFrame(train)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(x_data_il1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train), :] = train1

testPredictPlot = np.empty_like(x_data_il1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train1):len(x_data_il1), :] = predictions1

lower = np.empty_like(x_data_il1)
lower[:, :] = np.nan
lower[len(train1):len(x_data_il1), :] = low

upper = np.empty_like(x_data_il1)
upper[:, :] = np.nan
upper[len(train1):len(x_data_il1), :] = up
testpp = pd.DataFrame(testPredictPlot)
trainpp = pd.DataFrame(trainPredictPlot)
df2 = pd.DataFrame(il1nxtweek['DeviceTimeStamp'])
df2 = df2.reset_index()
df2 = df2.drop(['index'], axis=1)
training = df2.merge(trainpp, left_index=True, right_index=True)
predicted = df2.merge(testpp, left_index=True, right_index=True)
original1_il1 = il1nxtweek
original1_il1 = original1_il1.set_index('DeviceTimeStamp')
trn_il1 = training
trn_il1 = trn_il1.set_index('DeviceTimeStamp')
prd_il1 = predicted
prd_il1 = prd_il1.set_index('DeviceTimeStamp')
fault_pred_il1 = prd_il1[prd_il1[0] > 1.2 * 180]
il1_faults_next_day = len(fault_pred_il1)
s_il1 = ''
if (il1_faults_next_day > 0):
    s_il1 = 'Fault may occur next 24hrs'
else:
    s_il1 = 'Fault may not occur next 24hrs'
upb = pd.DataFrame(upper)
ub_il1 = df2.merge(upb, left_index=True, right_index=True)
ub_il1 = ub_il1.set_index('DeviceTimeStamp')
lwb = pd.DataFrame(lower)
lb_il1 = df2.merge(lwb, left_index=True, right_index=True)
lb_il1 = lb_il1.set_index('DeviceTimeStamp')

# IL2

il2nxtweek = cv[cv['DeviceTimeStamp'] >= week]
il2nxtweek_high = il2nxtweek.query('IL2>=1.2*180', inplace=False)
total_il2nxtwk_high = len(il2nxtweek_high)
cv_il2_high = cv.query('IL2>=1.2*180', inplace=False)
total_il2_high = len(cv_il2_high)
# ARIMA FOR IL2
dataset_il2 = il2nxtweek['IL2'].values
dataset_il2 = dataset_il2.astype('float32')
# fit model
model_il2 = ARIMA(il2nxtweek['IL2'], order=(5, 1, 0))
model_fit_il2 = model_il2.fit(disp=0)
X = il2nxtweek['IL2'].values
size = int(len(X) * 6 / 7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
confidence = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output, error, conf = model_fit.forecast()
    yhat = output[0]
    predictions.append([yhat, conf[0][0], conf[0][1]])
    obs = test[t]
    history.append(obs)
pred = pd.DataFrame(predictions)
error = mean_squared_error(test, pred[0])
rmse_il2 = np.sqrt(mean_squared_error(test, pred[0]))
x_data_il2 = pd.DataFrame(il2nxtweek['IL2'])
test1 = test.reshape(-1, 1)
test1 = pd.DataFrame(test1)
predictions1 = pd.DataFrame(pred[0])
low = pd.DataFrame(pred[1])
up = pd.DataFrame(pred[2])
train1 = pd.DataFrame(train)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(x_data_il2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train), :] = train1
testPredictPlot = np.empty_like(x_data_il2)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train1):len(x_data_il2), :] = predictions1
lower = np.empty_like(x_data_il2)
lower[:, :] = np.nan
lower[len(train1):len(x_data_il2), :] = low
upper = np.empty_like(x_data_il2)
upper[:, :] = np.nan
upper[len(train1):len(x_data_il2), :] = up
testpp = pd.DataFrame(testPredictPlot)
trainpp = pd.DataFrame(trainPredictPlot)
df2 = pd.DataFrame(il2nxtweek['DeviceTimeStamp'])
df2 = df2.reset_index()
df2 = df2.drop(['index'], axis=1)
training = df2.merge(trainpp, left_index=True, right_index=True)
predicted = df2.merge(testpp, left_index=True, right_index=True)
original1_il2 = il2nxtweek
original1_il2 = original1_il2.set_index('DeviceTimeStamp')
trn_il2 = training
trn_il2 = trn_il2.set_index('DeviceTimeStamp')
prd_il2 = predicted
prd_il2 = prd_il2.set_index('DeviceTimeStamp')
fault_pred_il2 = prd_il2[prd_il2[0] > 1.2 * 180]
il2_faults_next_day = len(fault_pred_il2)
s_il2 = ''
if (il2_faults_next_day > 0):
    s_il2 = 'Fault may occur next 24hrs'
else:
    s_il2 = 'Fault may not occur next 24hrs'
upb = pd.DataFrame(upper)
ub_il2 = df2.merge(upb, left_index=True, right_index=True)
ub_il2 = ub_il2.set_index('DeviceTimeStamp')
lwb = pd.DataFrame(lower)
lb_il2 = df2.merge(lwb, left_index=True, right_index=True)
lb_il2 = lb_il2.set_index('DeviceTimeStamp')

# IL3

il3nxtweek = cv[cv['DeviceTimeStamp'] >= week]
il3nxtweek_high = il3nxtweek.query('IL3>=1.2*180', inplace=False)
total_il3nxtwk_high = len(il3nxtweek_high)
cv_il3_high = cv.query('IL3>=1.2*180', inplace=False)
total_il3_high = len(cv_il3_high)

# ARIMA of IL3
dataset_il3 = il3nxtweek['IL3'].values
dataset_il3 = dataset_il3.astype('float32')

# fit model
model_il3 = ARIMA(il3nxtweek['IL3'], order=(5, 1, 0))
model_fit_il3 = model_il3.fit(disp=0)

X = il3nxtweek['IL3'].values
size = int(len(X) * 6 / 7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
confidence = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output, error, conf = model_fit.forecast()
    yhat = output[0]
    predictions.append([yhat, conf[0][0], conf[0][1]])
    obs = test[t]
    history.append(obs)
pred = pd.DataFrame(predictions)
error = mean_squared_error(test, pred[0])
rmse_il3 = np.sqrt(mean_squared_error(test, pred[0]))
x_data_il3 = pd.DataFrame(il3nxtweek['IL3'])
test1 = test.reshape(-1, 1)
test1 = pd.DataFrame(test1)
predictions1 = pd.DataFrame(pred[0])
low = pd.DataFrame(pred[1])
up = pd.DataFrame(pred[2])
train1 = pd.DataFrame(train)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(x_data_il3)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train), :] = train1

testPredictPlot = np.empty_like(x_data_il3)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train1):len(x_data_il3), :] = predictions1

lower = np.empty_like(x_data_il3)
lower[:, :] = np.nan
lower[len(train1):len(x_data_il3), :] = low

upper = np.empty_like(x_data_il3)
upper[:, :] = np.nan
upper[len(train1):len(x_data_il3), :] = up
testpp = pd.DataFrame(testPredictPlot)
trainpp = pd.DataFrame(trainPredictPlot)
df2 = pd.DataFrame(il3nxtweek['DeviceTimeStamp'])
df2 = df2.reset_index()
df2 = df2.drop(['index'], axis=1)
training = df2.merge(trainpp, left_index=True, right_index=True)
predicted = df2.merge(testpp, left_index=True, right_index=True)
original1_il3 = il3nxtweek
original1_il3 = original1_il3.set_index('DeviceTimeStamp')
trn_il3 = training
trn_il3 = trn_il3.set_index('DeviceTimeStamp')
prd_il3 = predicted
prd_il3 = prd_il3.set_index('DeviceTimeStamp')
fault_pred_il3 = prd_il3[prd_il3[0] > 1.2 * 180]
il3_faults_next_day = len(fault_pred_il3)
s_il3 = ''
if (il3_faults_next_day > 0):
    s_il3 = 'Fault may occur next 24hrs'
else:
    s_il3 = 'Fault may not occur next 24hrs'
upb = pd.DataFrame(upper)
ub_il3 = df2.merge(upb, left_index=True, right_index=True)
ub_il3 = ub_il3.set_index('DeviceTimeStamp')
lwb = pd.DataFrame(lower)
lb_il3 = df2.merge(lwb, left_index=True, right_index=True)
lb_il3 = lb_il3.set_index('DeviceTimeStamp')

url1 = "http://182.18.164.19:3402/Trans/TotalPower/868997035786613/all"
user = "admin"
passwd = "admin@123"
auth_values = (user, passwd)
response1 = requests.get(url1, auth=auth_values)
dat = response1.json()
tp1 = pd.DataFrame(dat)
tp1 = tp1.iloc[::-1]
tp1.drop_duplicates(subset="DeviceTimeStamp", keep=False, inplace=True)
tp = tp1.reset_index()
tp = tp.drop(['index'], axis=1)
tp['DeviceTimeStamp'] = pd.to_datetime(tp['DeviceTimeStamp'])
x = tp['DeviceTimeStamp'].max()
y = tp['DeviceTimeStamp'].min()
days = datetime.timedelta(7)
week = x - days
query1 = tp[tp['DeviceTimeStamp'] >= week]
dataset = query1['KW'].values
dataset = dataset.astype('float32')
model = ARIMA(query1['KW'], order=(5, 1, 0))
model_fit = model.fit(disp=0)
X = query1['KW'].values
size = int(len(X) * 6 / 7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
confidence = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output, error, conf = model_fit.forecast()
    yhat = output[0]
    predictions.append([yhat, conf[0][0], conf[0][1]])
    obs = test[t]
    history.append(obs)
pred = pd.DataFrame(predictions)
error = mean_squared_error(test, pred[0])
rmse = np.sqrt(mean_squared_error(test, pred[0]))
x_data = pd.DataFrame(query1['KW'])
test1 = test.reshape(-1, 1)
test1 = pd.DataFrame(test1)
predictions1 = pd.DataFrame(pred[0])
low = pd.DataFrame(pred[1])
up = pd.DataFrame(pred[2])
train1 = pd.DataFrame(train)
trainPredictPlot = np.empty_like(x_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train), :] = train1
testPredictPlot = np.empty_like(x_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train1):len(x_data), :] = predictions1
lower = np.empty_like(x_data)
lower[:, :] = np.nan
lower[len(train1):len(x_data), :] = low
upper = np.empty_like(x_data)
upper[:, :] = np.nan
upper[len(train1):len(x_data), :] = up
testpp = pd.DataFrame(testPredictPlot)
trainpp = pd.DataFrame(trainPredictPlot)
df2 = pd.DataFrame(query1['DeviceTimeStamp'])
df2 = df2.reset_index()
df2 = df2.drop(['index'], axis=1)
training = df2.merge(trainpp, left_index=True, right_index=True)
predicted = df2.merge(testpp, left_index=True, right_index=True)
original1 = query1
original1 = original1.set_index('DeviceTimeStamp')
trn = training
trn = trn.set_index('DeviceTimeStamp')
prd = predicted
prd = prd.set_index('DeviceTimeStamp')
upb = pd.DataFrame(upper)
ub = df2.merge(upb, left_index=True, right_index=True)
ub = ub.set_index('DeviceTimeStamp')
lwb = pd.DataFrame(lower)
lb = df2.merge(lwb, left_index=True, right_index=True)
lb = lb.set_index('DeviceTimeStamp')

url = "http://182.18.164.19:3402/Trans/CurrentVoltage/868997035786613/all"
user = "admin"
passwd = "admin@123"
auth_values = (user, passwd)
response = requests.get(url, auth=auth_values)
lis = response.json()
head = list(lis[0].keys())
rawdata = []
for i in range(0, len(lis)):
    lim = []
    for x in head:
        if x == 'DeviceImei' or x == 'A_id':
            lim.append(lis[i][x])
        elif x == 'DeviceTimeStamp':
            s = lis[i][x]
            lim.append(s[0:16])
        else:
            lim.append(float(lis[i][x]))
    rawdata.append(lim)


def rev(lst):
    return [ele for ele in reversed(lst)]


data = rev(rawdata)
cv = pd.DataFrame(data, columns=head)
cv['DeviceTimeStamp'] = pd.to_datetime(cv['DeviceTimeStamp'])


# Create your views here.

def home(request):
    plot_data = [
        go.Scatter(
            x=original1.index,
            y=original1['KW'],
            name='KW'
        ),
        go.Scatter(
            x=prd.index,
            y=prd[0],
            name='Predicted'
        ),
        go.Scatter(
            x=ub.index,
            y=ub[0],
            name='Upper Bound'
        ),
        go.Scatter(
            x=trn.index,
            y=trn[0],
            name='Training'
        ),
        go.Scatter(
            x=lb.index,
            y=lb[0],
            name='Lower Bound'
        )

    ]
    plot_layout = go.Layout(
        title='Load Prediction Using Arima, rmse=' + str(rmse),
        yaxis_title='IL2',
        xaxis_title='Time'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    fault_div = pyoff.plot(fig, auto_open=False, output_type="div")
    df = pd.read_csv('traffic/overviewcsv.csv')
    plot_data = [
        go.Scatter(
            x=df['DeviceTimeStamp'],
            y=df['OTI'],
            name='OTI'
        ),
        go.Scatter(
            x=df['DeviceTimeStamp'],
            y=df['ATI'],
            name='ATI'
        ),
        go.Scatter(
            x=df['DeviceTimeStamp'],
            y=df['OLI'],
            name='OLI'
        ),
    ]
    plot_layout = go.Layout(
        title='OTI Vs Time',
        yaxis_title='OTI',
        xaxis_title='Time'
    )
    fig1 = go.Figure(data=plot_data, layout=plot_layout)
    fault_div1 = pyoff.plot(fig1, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['VL12'],
            name='VL12'
        ),
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['VL23'],
            name='VL23'
        ),
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['VL31'],
            name='VL31'
        )

    ]
    plot_layout = go.Layout(
        title='Line Voltages, (ALL time)',
        yaxis_title='Line Voltages',
        xaxis_title='Time'
    )
    fig2 = go.Figure(data=plot_data, layout=plot_layout)
    fault_div2 = pyoff.plot(fig2, auto_open=False, output_type="div")

    return render(request, 'traffic/graphh.html',
                  {'fault_div': fault_div, 'fault_div1': fault_div1, 'fault_div2': fault_div2})


def fault_view(request):
    plot_data = [
        go.Scatter(
            x=original1_inut.index,
            y=original1_inut['INUT'],
            name='INUT'
        ),
        go.Scatter(
            x=prd_inut.index,
            y=prd_inut[0],
            name='Predicted'
        ),
        go.Scatter(
            x=ub_inut.index,
            y=ub_inut[0],
            name='Upper Bound'
        ),
        go.Scatter(
            x=trn_inut.index,
            y=trn_inut[0],
            name='Training'
        ),
        go.Scatter(
            x=lb_inut.index,
            y=lb_inut[0],
            name='Lower Bound'
        )

    ]
    plot_layout = go.Layout(
        title='INUT, rmse=' + str(rmse_inut) + ', ' + s_inut,
        yaxis_title='INUT',
        xaxis_title='Time'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    graph_div = pyoff.plot(fig, auto_open=False, output_type="div")

    plot_data = [
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['INUT'],
            name='INUT'
        ),
        go.Scatter(
            x=cv_inut_high['DeviceTimeStamp'],
            y=cv_inut_high['INUT'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='INUT (All time), Alarm count = ' + str(total_inut_high)
    )
    fig2 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div2 = pyoff.plot(fig2, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=inutnxtweek['DeviceTimeStamp'],
            y=inutnxtweek['INUT'],
            name='INUT'
        ),
        go.Scatter(
            x=inutnxtweek_high['DeviceTimeStamp'],
            y=inutnxtweek_high['INUT'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='INUT (All time), Alarm count last week = ' + str(total_inutnxtwk_high)
    )
    fig1 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div1 = pyoff.plot(fig1, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['IL1'],
            name='IL1'
        ),
        go.Scatter(
            x=cv_il1_high['DeviceTimeStamp'],
            y=cv_il1_high['IL1'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL1 (All time), Alarm count = ' + str(total_il1_high)
    )
    fig3 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div3 = pyoff.plot(fig3, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=il1nxtweek['DeviceTimeStamp'],
            y=il1nxtweek['IL1'],
            name='IL1'
        ),
        go.Scatter(
            x=il1nxtweek_high['DeviceTimeStamp'],
            y=il1nxtweek_high['IL1'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL1 (All time), Alarm count last week = ' + str(total_il1nxtwk_high)
    )
    fig4 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div4 = pyoff.plot(fig4, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=original1_il1.index,
            y=original1_il1['IL1'],
            name='IL1'
        ),
        go.Scatter(
            x=prd_il1.index,
            y=prd_il1[0],
            name='Predicted'
        ),
        go.Scatter(
            x=ub_il1.index,
            y=ub_il1[0],
            name='Upper Bound'
        ),
        go.Scatter(
            x=trn_il1.index,
            y=trn_il1[0],
            name='Training'
        ),
        go.Scatter(
            x=lb_il1.index,
            y=lb_il1[0],
            name='Lower Bound'
        )

    ]
    plot_layout = go.Layout(
        title='Load, rmse=' + str(rmse_il1) + ', ' + s_il1,
        yaxis_title='IL1',
        xaxis_title='Time'
    )
    fig5 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div5 = pyoff.plot(fig5, auto_open=False, output_type="div")

    plot_data = [
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['IL2'],
            name='IL2'
        ),
        go.Scatter(
            x=cv_il2_high['DeviceTimeStamp'],
            y=cv_il2_high['IL2'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL2 (All time), Alarm count = ' + str(total_il2_high)
    )
    fig6 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div6 = pyoff.plot(fig6, auto_open=False, output_type="div")

    plot_data = [
        go.Scatter(
            x=il2nxtweek['DeviceTimeStamp'],
            y=il2nxtweek['IL2'],
            name='IL2'
        ),
        go.Scatter(
            x=il2nxtweek_high['DeviceTimeStamp'],
            y=il2nxtweek_high['IL2'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL2 (All time), Alarm count last week = ' + str(total_il2nxtwk_high)
    )
    fig7 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div7 = pyoff.plot(fig7, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=original1_il2.index,
            y=original1_il2['IL2'],
            name='IL2'
        ),
        go.Scatter(
            x=prd_il2.index,
            y=prd_il2[0],
            name='Predicted'
        ),
        go.Scatter(
            x=ub_il2.index,
            y=ub_il2[0],
            name='Upper Bound'
        ),
        go.Scatter(
            x=trn_il2.index,
            y=trn_il2[0],
            name='Training'
        ),
        go.Scatter(
            x=lb_il2.index,
            y=lb_il2[0],
            name='Lower Bound'
        )

    ]
    plot_layout = go.Layout(
        title='IL2, rmse=' + str(rmse_il2) + ', ' + s_il2,
        yaxis_title='IL2',
        xaxis_title='Time'
    )
    fig8 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div8 = pyoff.plot(fig8, auto_open=False, output_type="div")

    plot_data = [
        go.Scatter(
            x=cv['DeviceTimeStamp'],
            y=cv['IL3'],
            name='IL3'
        ),
        go.Scatter(
            x=cv_il3_high['DeviceTimeStamp'],
            y=cv_il3_high['IL3'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL3 (All time), Alarm count = ' + str(total_il3_high)
    )
    fig9 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div9 = pyoff.plot(fig9, auto_open=False, output_type="div")
    plot_data = [
        go.Scatter(
            x=il3nxtweek['DeviceTimeStamp'],
            y=il3nxtweek['IL3'],
            name='IL3'
        ),
        go.Scatter(
            x=il3nxtweek_high['DeviceTimeStamp'],
            y=il3nxtweek_high['IL3'],
            mode='markers',
            name='Alarm',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='red',
                        opacity=0.8
                        )
        )

    ]
    plot_layout = go.Layout(
        title='IL3 (Last Week), Alarm count last week = ' + str(total_il3nxtwk_high)
    )
    fig10 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div10 = pyoff.plot(fig10, auto_open=False, output_type="div")

    plot_data = [
        go.Scatter(
            x=original1_il3.index,
            y=original1_il3['IL3'],
            name='IL3'
        ),
        go.Scatter(
            x=prd_il3.index,
            y=prd_il3[0],
            name='Predicted'
        ),
        go.Scatter(
            x=ub_il3.index,
            y=ub_il3[0],
            name='Upper Bound'
        ),
        go.Scatter(
            x=trn_il3.index,
            y=trn_il3[0],
            name='Training'
        ),
        go.Scatter(
            x=lb_il3.index,
            y=lb_il3[0],
            name='Lower Bound'
        )

    ]
    plot_layout = go.Layout(
        title='IL3, rmse=' + str(rmse_il3) + ', ' + s_il3,
        yaxis_title='IL3',
        xaxis_title='Time'
    )
    fig11 = go.Figure(data=plot_data, layout=plot_layout)
    graph_div11 = pyoff.plot(fig11, auto_open=False, output_type="div")

    return render(request, 'traffic/graph.html',
                  {'graph_div': graph_div, 'graph_div1': graph_div1, 'graph_div2': graph_div2,
                   'graph_div3': graph_div3, 'graph_div4': graph_div4, 'graph_div5': graph_div5,
                   'graph_div6': graph_div6, 'graph_div7': graph_div7, 'graph_div8': graph_div8,
                   'graph_div9': graph_div9, 'graph_div10': graph_div10, 'graph_div11': graph_div11})

def file(request):
    return render(request,'nav.html')