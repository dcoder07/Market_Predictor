import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 

testIndex = yf.Ticker("NVDA")
testIndex = testIndex.history(period="max")

# testIndex.plot(y="Close", use_index=True)
# plt.show()

del testIndex["Dividends"]
del testIndex["Stock Splits"]

testIndex["NextDayClose"] = testIndex["Close"].shift(-1)
testIndex["Target"] = (testIndex["NextDayClose"] > testIndex["Close"]).astype(int)

testIndex = testIndex.loc["2000-01-01":].copy()

# print(testIndex)
        
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = testIndex.iloc[:-100]    
test = testIndex.iloc[-100:]    

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

predictions = backtest(testIndex,model,predictors)

# noOfPredictions = predictions["Predictions"].value_counts() 
# print(noOfPredictions) 

# predPercentage = (predictions["Predictions"].value_counts()/predictions.shape[0])*100 
# print(predPercentage) 

predictScore = precision_score(predictions["Target"],predictions["Predictions"])
print("Precision Score: ",predictScore)

horizons = [2,5,250,1000]
new_predictors = []

for hor in horizons:
    rolling_avg = testIndex.rolling(hor).mean()

    ratio_column = f"Close_Ratio_{hor}"
    testIndex[ratio_column] = testIndex["Close"]/rolling_avg["Close"]

    trend_column = f"Trend_{hor}"
    testIndex[trend_column] = testIndex.shift(1).rolling(hor).sum()["Target"]
    
    new_predictors+=[ratio_column,trend_column]

# testIndex = testIndex.dropna()

new_predictions = backtest(testIndex,model,new_predictors)

precisionScore = precision_score(new_predictions["Target"],new_predictions["Predictions"]) 
print("Precison Score: ",precisionScore)

noOfNewPredictions = new_predictions["Predictions"].value_counts()
print(noOfNewPredictions)

print(testIndex)
