from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi import HTTPException

app = FastAPI()

model = joblib.load("Model/best_model.pkl")

class Features(BaseModel):
    Date: int
    Open: float
    High: float
    Low: float
    Close: float
    Trades: float
    Volume: float

@app.get("/info", response_class=JSONResponse)
def info_check():
    return {
        "model_name": "Financial Prediction Model",
        "model_version": "1.0",
        "description": "Model koristi vremenske i trzisne podatke (DATE, O, H, L, C, T, V) za predikciju buduce vrijednosti povrata.",
        "input_requirements": {
            "Date": "Datum koji oznacava sat u kojem su izmjereni podaci. (milisekunde)",
            "Open": "Pocetna cijena sata (float)",
            "High": "Najvisa cijena sata (float)",
            "Low": "Najniza cijena sata (float)",
            "Close": "Zavrsna cijena sata (float)",
            "Trades": "Ukupan broj transakcija u satu (float)",
            "Volume": "Ukupan broj trade-anih jedinica instrumenta u satu (float)"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/predict",
            "input_format": {
                "features": {
                    "Date": "int",
                    "Open": "float",
                    "High": "float",
                    "Low": "float",
                    "Close": "float",
                    "Trades": "float",
                    "Volume": "float"
                }
            }
        }
    }

def compute_features(df_set: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data['Date'] = df_set['Date']
    df_set['return'] = df_set['Close'].pct_change()
    data['return'] = df_set['Close'].pct_change()
    data['return_t-1'] = df_set['return'].shift(1)
    data['return_t-5'] = df_set['return'].shift(5)
    data['return_t-10'] = df_set['return'].shift(10)
    data['volatility_10'] = df_set['return'].rolling(10).std()
    data['SMA_5'] = df_set['Close'].rolling(5).mean()
    data['SMA_20'] = df_set['Close'].rolling(20).mean()
    data['EMA_10'] = df_set['Close'].ewm(span=10, adjust=False).mean()
    data['volume_ind'] = df_set['Volume'] / df_set['Volume'].rolling(10).mean()
    data['H/L'] = df_set['High'] / df_set['Low']
    data['C/O'] = df_set['Close'] / df_set['Open']
    data['momentum_5'] = df_set['Close'] - df_set['Close'].shift(5)
    data['day_of_week'] = df_set['Date'].dt.dayofweek
    data['month'] = df_set['Date'].dt.month
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data = data.dropna().reset_index(drop=True)
    data = data.drop(columns=['Date','day_of_week','month','return'])

    cols_to_scale = ['return_t-1','return_t-5','return_t-10','volatility_10','SMA_5', 'SMA_20', 'EMA_10','volume_ind','H/L','C/O','momentum_5']
    scaler = StandardScaler()
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data

expected_format = {
    "Date": "int",
    "Open": "float",
    "High": "float",
    "Low": "float",
    "Close": "float",
    "Trades": "float",
    "Volume": "float"
}

@app.post("/predict")
def predict(new_data: Features):
    try:
        new_row =  {
            "Date": pd.to_datetime(new_data.Date, unit='ms'),
            "Open": new_data.Open,
            "High": new_data.High,
            "Low": new_data.Low,
            "Close": new_data.Close,
            "Trades": new_data.Trades,
            "Volume": new_data.Volume
        }
        df = pd.read_parquet("Data/btc_historical_price.parquet")
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df_subset= df.tail(100).copy()

        features_df = compute_features(df_subset)
        last_features = features_df.iloc[-1].values.reshape(1, -1)
        y_pred = model.predict(last_features)
        return {"Prediction": float(y_pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Greška u inputu: {e}")