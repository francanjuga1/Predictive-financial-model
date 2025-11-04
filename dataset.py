from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

def test_info(dataset):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    X = dataset.drop(columns=['Date', 'return'])  
    y = dataset['return']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_names = X.columns

    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def prepare_data(file_name):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(file_name)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close']) #uklonjene null vrijednosti

    df['Trades'] = df['Trades'].astype('int64') # Trades -> int64
    #print(df.describe())

    #z_scores = np.abs(stats.zscore(df[['Open','High','Low','Close','Trades','Volume']]))
    #outliers = (z_scores > 3)  # 3sigma
    #print(np.where(outliers))

    data= pd.DataFrame()
    data['Date'] = df['Date']
    df['return'] = df['Close'].pct_change()
    data['return'] = df['Close'].pct_change()
    data['return_t-1'] = df['return'].shift(1)
    data['return_t-5'] = df['return'].shift(5)
    data['return_t-10'] = df['return'].shift(10)
    data['volatility_10'] = df['return'].rolling(10).std()
    data['SMA_5'] = df['Close'].rolling(5).mean()
    data['SMA_20'] = df['Close'].rolling(20).mean()
    data['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    data['volume_ind'] = df['Volume'] / df['Volume'].rolling(10).mean()
    data['H/L'] = df['High'] / df['Low']
    data['C/O'] = df['Close'] / df['Open']
    data['momentum_5'] = df['Close'] - df['Close'].shift(5)
    data['day_of_week'] = df['Date'].dt.dayofweek
    data['month'] = df['Date'].dt.month

    data = data.dropna().reset_index(drop=True)

    #subset = data.head(500)
    #subset.to_csv("data_500rows.csv", index=False)

    #test_info(data)
    return_t1 = pd.DataFrame()
    return_t1['return_t+1'] = data['return'].shift(-1).dropna()
    data = data.drop(columns='return')
    #visualize(data)

    cols_to_scale = ['return_t-1','return_t-5','return_t-10','volatility_10','SMA_5', 'SMA_20', 'EMA_10','volume_ind','H/L','C/O','momentum_5']

    scaler = StandardScaler()

    # Skaliranje
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
    #subset = data.head(10000)
    #subset.to_csv("data_500rows.csv", index=False)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)  #dodavanje periodičkih varijabli
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data = data.drop(columns=['Date','day_of_week','month']) #dropanje Date, i vremenski varijabli
    data = data.iloc[:-1].reset_index(drop=True)# dropati zadnji red

    #podijela podataka za train\val\test
    n = len(data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    # podjela
    train_set = data.iloc[:train_end]
    val_set = data.iloc[train_end:val_end]
    test_set = data.iloc[val_end:]
    
    #return_t1.head(100).to_csv("return_t+1.csv", index=False)
    #train_set.head(100).to_csv("train.csv", index=False)
    #val_set.head(100).to_csv("val.csv", index=False)
    #test_set.head(100).to_csv("test.csv", index=False)

    return return_t1,train_set,val_set,test_set 

def visualize(data):
    sns.histplot(data['return_t-1'], bins=50, kde=True)
    plt.title(f"sigma = {math.sqrt(data['return_t-1'].var()):.3f}")
    plt.show()

    sns.histplot(data['return_t-5'], bins=50,color="green", kde=True)
    plt.title(f"sigma = {math.sqrt(data['return_t-5'].var()):.3f}")
    plt.show()

    sns.histplot(data['return_t-10'], bins=50,color="red", kde=True)
    plt.title(f"sigma = {math.sqrt(data['return_t-10'].var()):.3f}")
    plt.show()

    sns.histplot(data['SMA_5'], bins=50,color="blue", kde=True)
    plt.title(f"sigma = {math.sqrt(data['SMA_5'].var()):.3f}")
    plt.show()
    
    sns.histplot(data['SMA_20'], bins=50,color="red", kde=True)
    plt.title(f"sigma = {math.sqrt(data['SMA_20'].var()):.3f}")
    plt.show()

    sns.histplot(data['EMA_10'], bins=50,color="green", kde=True)
    plt.title(f"sigma = {math.sqrt(data['EMA_10'].var()):.3f}")
    plt.show()

    sns.histplot(data['volume_ind'], bins=50,color="green", kde=True)
    plt.title(f"sigma = {math.sqrt(data['volume_ind'].var()):.3f}")
    plt.show()
    
    sns.histplot(data['H/L'], bins=50,color="red", kde=True)
    plt.title(f"sigma = {math.sqrt(data['H/L'].var()):.3f}")
    plt.show()

    sns.histplot(data['C/O'], bins=50,color="blue", kde=True)
    plt.title(f"sigma = {math.sqrt(data['C/O'].var()):.3f}")
    plt.show()

    sns.histplot(data['momentum_5'], bins=50,color="yellow", kde=True)
    plt.title(f"sigma = {math.sqrt(data['momentum_5'].var()):.3f}")
    plt.show()

    data.drop(columns=['Date'])
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacijska matrica značajki")
    plt.show()