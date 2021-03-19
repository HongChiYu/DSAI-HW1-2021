if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    
    
    # create and evaluate an updated autoregressive model
    import pandas as pd
    from pandas import read_csv
    from matplotlib import pyplot
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from pandas.plotting import lag_plot
    from pandas.plotting import autocorrelation_plot
    from pandas import DataFrame
    from pandas import concat

    # load dataset
    series = read_csv("dataset.csv", header=0, index_col=0)
    series.plot()
    pyplot.show()
    
    lag_plot(series)
    pyplot.show()
    
    autocorrelation_plot(series)
    pyplot.show()
    
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print(result)
    
    # split dataset
    X = series.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    # train autoregression
    window = 30
    model = AutoReg(train, lags=30)
    model_fit = model.fit()
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(int(yhat))
    	history.append(obs)
    	print('predicted=%f, expected=%f' % (int(yhat), obs))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    
    # plot
    
    values = ['20210323', '20210324', '20210325', '20210326','20210327','20210328','20210329'] 
    pyplot.plot(values,test)
    pyplot.plot(values,predictions, color='red')
    pyplot.ylabel("Operating Reserve(MW)")
    pyplot.xticks(values)
    pyplot.show()
    
    
    df1 = pd.DataFrame(values,columns=['date'])
    df2 = pd.DataFrame(predictions,columns=['operating_reserve(MW)'])
    df = pd.concat([df1, df2], axis=1)
    df.to_csv('submission.csv',index=False)
