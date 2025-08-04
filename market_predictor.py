import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 

sp500 = yf.Ticker("^BSESN")
sp500 = sp500.history(period="max")

# sp500.plot(y="Close", use_index=True)
# plt.show()

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["NextDayClose"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["NextDayClose"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["2000-01-01":].copy()

# print(sp500)
        
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]    
test = sp500.iloc[-100:]    

predictors = ["Open", "High", "Low", "Close", "Volume"]
model.fit(train[predictors],train["Target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds,index=test.index)   

# print(preds)

score = precision_score(test["Target"],preds)

# print(score)    


def predict(test,train,model,predictors):
    model.fit(train[predictors],train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6] = 1
    preds[preds<.6] = 0
    preds = pd.Series(preds,index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined


def backtest(data,model,predictors,start=2500,step=250):
    all_predictions = []
    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(test,train,model,predictors)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500,model,predictors)

# noOfPredictions = predictions["Predictions"].value_counts() 
# print(noOfPredictions) 

# predPercentage = (predictions["Predictions"].value_counts()/predictions.shape[0])*100 
# print(predPercentage) 

predictScore = precision_score(predictions["Target"],predictions["Predictions"])
print("Precision Score: ",predictScore)

horizons = [2,5,250,1000]
new_predictors = []

for hor in horizons:
    rolling_avg = sp500.rolling(hor).mean()

    ratio_column = f"Close_Ratio_{hor}"
    sp500[ratio_column] = sp500["Close"]/rolling_avg["Close"]

    trend_column = f"Trend_{hor}"
    sp500[trend_column] = sp500.shift(1).rolling(hor).sum()["Target"]
    
    new_predictors+=[ratio_column,trend_column]

# sp500 = sp500.dropna()

new_predictions = backtest(sp500,model,new_predictors)

precisionScore = precision_score(new_predictions["Target"],new_predictions["Predictions"]) 
print("Precison Score: ",precisionScore)

noOfNewPredictions = new_predictions["Predictions"].value_counts()
print(noOfNewPredictions)

print(sp500)
