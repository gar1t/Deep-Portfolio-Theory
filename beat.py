import copy
import warnings

from collections import defaultdict

warnings.simplefilter("ignore", FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter("ignore", DataConversionWarning)

encode_epochs = 500
encode_batch_size = 10
calibrate_epochs = 500
calibrate_batch_size = 10

###################################################################
# Load data
###################################################################

# stock componenet data
stock = defaultdict(defaultdict)

stock_lp = pd.read_csv('data/last_price.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
stock['calibrate']['lp'] = stock_lp.iloc[0:104, :]
stock['validate']['lp'] = stock_lp.iloc[104:, :]

stock_net = pd.read_csv('data/net_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
stock['calibrate']['net'] = stock_net.iloc[0:104, :]
stock['validate']['net'] = stock_net.iloc[104:, :]

stock_percentage = pd.read_csv('data/percentage_change.csv', index_col=0).dropna(axis=1, how='any').astype('float32')
stock['calibrate']['percentage'] = stock_percentage.iloc[0:104, :]
stock['validate']['percentage'] = stock_percentage.iloc[104:, :]


# ibb data
ibb = defaultdict(defaultdict)
ibb_full = pd.read_csv('data/ibb.csv', index_col=0).astype('float32')

ibb_lp = ibb_full.iloc[:,0] # Series
ibb['calibrate']['lp'] = ibb_lp[0:104]
ibb['validate']['lp'] = ibb_lp[104:]

ibb_net = ibb_full.iloc[:,1] # Series
ibb['calibrate']['net'] = ibb_net[0:104]
ibb['validate']['net'] = ibb_net[104:]

ibb_percentage = ibb_full.iloc[:,2] # Series
ibb['calibrate']['percentage'] = ibb_percentage[0:104]
ibb['validate']['percentage'] = ibb_percentage[104:]

###################################################################
# Encoding
###################################################################

encoding_dim = 5 # 5 neurons
num_stock = len(stock_lp.columns) # Use 83 stocks as features

# connect all layers (see 'Stacked Auto-Encoders' in paper)
input_img = Input(shape=(num_stock, ))
encoded = Dense(
    encoding_dim,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.01))(input_img)
decoded = Dense(
    num_stock,
    activation='linear',
    kernel_regularizer=regularizers.l2(0.01))(encoded)

# construct and compile AE model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

# train autoencoder
data = stock['calibrate']['net']
autoencoder.fit(
    data,
    data,
    shuffle=False,
    epochs=encode_epochs,
    batch_size=encode_batch_size)
autoencoder.save('model/beat_autoencoder.h5')

# test/reconstruct market information matrix
reconstruct = autoencoder.predict(data)

communal_information = []

for i in range(0,83):
    # 2 norm difference
    diff = np.linalg.norm((data.iloc[:,i] - reconstruct[:,i]))
    communal_information.append(float(diff))

print("stock #, 2-norm, stock name")
ranking = np.array(communal_information).argsort()
for stock_index in ranking:
    # print stock name from lowest different to highest
    print(
        stock_index,
        communal_information[stock_index],
        stock['calibrate']['net'].iloc[:,stock_index].name)

which_stock = 65

# now decoded last price plot
stock_autoencoder = copy.deepcopy(reconstruct[:, which_stock])
stock_autoencoder[0] = 0
stock_autoencoder = stock_autoencoder.cumsum()
stock_autoencoder += (stock['calibrate']['lp'].iloc[0, which_stock])

## plot for comparison
pd.Series(
    stock['calibrate']['lp'].iloc[:, which_stock].as_matrix(),
    index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(
        label='stock original', legend=True)
pd.Series(
    stock_autoencoder,
    index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(
        label='stock autoencoded', legend=True)
plt.savefig("plots/original-vs-encoded.png")
plt.clf()

###################################################################
# Calibrate
###################################################################

# from -5% to 5%
y_amended = ibb['calibrate']['percentage']
y_amended[y_amended < -5] = 5

# re-calculate the last price
y_amended[0] = 0
relative_percentage = (y_amended /100) + 1
lp_amended = ibb['calibrate']['lp'][0] * (relative_percentage.cumprod())

# plot comparison
pd.Series(
    ibb['calibrate']['lp'].as_matrix(),
    index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(
        label='ibb original', legend=True)
pd.Series(
    lp_amended.as_matrix(),
    index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(
        label='ibb amended', legend=True)
plt.savefig("plots/ibb-original-vs-amended.png")
plt.clf()

ibb_predict = defaultdict(defaultdict)
total_2_norm_diff = defaultdict(defaultdict)
dl_scaler = defaultdict(StandardScaler)

for non_communal in [15, 35, 55]:
    # some numerical values
    encoding_dim = 5
    s = 10 + non_communal
    # portfolio index
    stock_index = np.concatenate((ranking[0:10], ranking[-non_communal:]))

    # connect all layers
    input_img = Input(shape=(s,))
    encoded = Dense(
        encoding_dim,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.005))(input_img)
    decoded = Dense(
        1,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.005))(encoded)

    # construct and compile deep learning routine
    deep_learner = Model(input_img, decoded)
    deep_learner.compile(optimizer='sgd', loss='mean_squared_error')

    x = stock['calibrate']['percentage'].iloc[:, stock_index]
    y = y_amended # amended percentage

    # Multi-layer Perceptron is sensitive to feature scaling, so it is
    # highly recommended to scale your data
    dl_scaler[s] = StandardScaler()
    dl_scaler[s].fit(x)
    x = dl_scaler[s].transform(x)

    deep_learner.fit(
        x,
        y,
        shuffle=False,
        epochs=calibrate_epochs,
        batch_size=calibrate_batch_size)
    deep_learner.save('model/beat_s' + str(s) + '.h5')

    # is it good?
    relative_percentage = copy.deepcopy(deep_learner.predict(x))
    relative_percentage[0] = 0
    relative_percentage = (relative_percentage/100) + 1

    ibb_predict['calibrate'][s] = (
        ibb['calibrate']['lp'][0] * (relative_percentage.cumprod()))
    # compare with amended last price
    total_2_norm_diff['calibrate'][s] = (
        np.linalg.norm((ibb_predict['calibrate'][s] - lp_amended)))

# plot results and 2-norm differences
pd.Series(
    ibb['calibrate']['lp'].as_matrix(),
    index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(
        label='ibb original', legend=True)
pd.Series(
    lp_amended.as_matrix(),
    index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(
        label='IBB Amended', legend=True)

for s in [25, 45, 65]:
    pd.Series(
        ibb_predict['calibrate'][s],
        index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(
            label='IBB S'+str(s), legend=True)
    print(
        "S" +str(s) + " 2-norm difference: ",
        total_2_norm_diff['calibrate'][s])
plt.savefig("plots/portfolio.performance.png")
plt.clf()

###################################################################
# Validate
###################################################################

for non_communal in [15, 35, 55]:
    # some numerical values
    encoding_dim = 5
    s = 10 + non_communal
    stock_index = np.concatenate((ranking[0:10], ranking[-non_communal:]))

    # load our trained models
    deep_learner = load_model('model/beat_s' + str(s) + '.h5')

    x = stock['validate']['percentage'].iloc[:, stock_index]
    x = dl_scaler[s].transform(x)

    # is it good?
    relative_percentage = copy.deepcopy(deep_learner.predict(x))
    relative_percentage[0] = 0
    relative_percentage = (relative_percentage/100) + 1

    ibb_predict['validate'][s] = (
        ibb['validate']['lp'][0] * (relative_percentage.cumprod()))

# plot original IBB last price
pd.Series(
    ibb['validate']['lp'].as_matrix(),
    index=pd.date_range(start='01/03/2014', periods=122, freq='W')).plot(
        label='IBB original', legend=True)

# 2-norm difference is now meaningless to compare
for s in [25, 45, 65]:
    pd.Series(
        ibb_predict['validate'][s],
        index=pd.date_range(start='01/03/2014', periods = 122,freq='W')).plot(
            label='IBB S'+str(s), legend=True)
plt.savefig("plots/portfolio.performance-test-dates.png")
plt.clf()
