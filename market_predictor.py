import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 

spBSE = yf.Ticker("^BSESN")
spBSE = spBSE.history(period="max")

# spBSE.plot(y="Close", use_index=True)
# plt.show()

del spBSE["Dividends"]
del spBSE["Stock Splits"]

spBSE["NextDayClose"] = spBSE["Close"].shift(-1)
spBSE["Target"] = (spBSE["NextDayClose"] > spBSE["Close"]).astype(int)

spBSE = spBSE.loc["2000-01-01":].copy()

# print(spBSE)
        
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = spBSE.iloc[:-100]    
test = spBSE.iloc[-100:]    

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

predictions = backtest(spBSE,model,predictors)

# noOfPredictions = predictions["Predictions"].value_counts() 
# print(noOfPredictions) 

# predPercentage = (predictions["Predictions"].value_counts()/predictions.shape[0])*100 
# print(predPercentage) 

predictScore = precision_score(predictions["Target"],predictions["Predictions"])
print("Precision Score: ",predictScore)

horizons = [2,5,250,1000]
new_predictors = []

for hor in horizons:
    rolling_avg = spBSE.rolling(hor).mean()

    ratio_column = f"Close_Ratio_{hor}"
    spBSE[ratio_column] = spBSE["Close"]/rolling_avg["Close"]

    trend_column = f"Trend_{hor}"
    spBSE[trend_column] = spBSE.shift(1).rolling(hor).sum()["Target"]
    
    new_predictors+=[ratio_column,trend_column]

# spBSE = spBSE.dropna()

new_predictions = backtest(spBSE,model,new_predictors)

precisionScore = precision_score(new_predictions["Target"],new_predictions["Predictions"]) 
print("Precison Score: ",precisionScore)

noOfNewPredictions = new_predictions["Predictions"].value_counts()
print(noOfNewPredictions)

print(spBSE)
