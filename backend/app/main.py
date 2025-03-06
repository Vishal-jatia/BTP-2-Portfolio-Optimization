from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from yahooquery import search
from fastapi.responses import JSONResponse
import yfinance as yf
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from typing import Dict, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TickerRequest(BaseModel):
    tickers: list[str]

def fetch_historical_data(tickers):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        history = stock.history(period="5y")
        data[ticker] = history['Close'].values
    return data

def train_ann(data):
    models = {}
    for ticker, prices in data.items():
        X, y = [], []
        for i in range(len(prices) - 10):
            X.append(prices[i:i+10])
            y.append(prices[i+10])
        X, y = np.array(X), np.array(y)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)
        models[ticker] = model
    return models

def predict_future_prices(models, data, duration="1y"):
    future_prices = {}
    duration_map = {"6m": 180, "1y": 365, "5y": 1825, "10y": 3650}
    prediction_days = duration_map.get(duration, 365)

    for ticker, model in models.items():
        last_10_days = np.array(data[ticker][-10:]).reshape(1, -1)
        future_preds = []

        for _ in range(prediction_days):
            predicted_price = model.predict(last_10_days)[0][0]
            future_preds.append(predicted_price)
            last_10_days = np.roll(last_10_days, -1)
            last_10_days[0, -1] = predicted_price

        future_prices[ticker] = future_preds

    return future_prices

class PortfolioOptimization(Problem):
    def __init__(self, future_prices):
        super().__init__(n_var=len(future_prices), n_obj=2, xl=0, xu=1)
        self.tickers = list(future_prices.keys())
        self.returns = np.array([np.mean(future_prices[t]) for t in self.tickers])
        self.risks = np.array([np.std(future_prices[t]) for t in self.tickers])

        scaler = MinMaxScaler()
        self.returns = scaler.fit_transform(self.returns.reshape(-1, 1)).flatten()
        self.risks = scaler.fit_transform(self.risks.reshape(-1, 1)).flatten()

    def _evaluate(self, X, out, *args, **kwargs):
        returns = np.sum(X * self.returns, axis=1)
        risks = np.sum(X * self.risks, axis=1)
        out["F"] = np.column_stack([-returns, risks])

def optimize_portfolio(future_prices, algorithm_type):
    problem = PortfolioOptimization(future_prices)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    
    algorithm = {
        "nsga2": NSGA2(),
        "moead": MOEAD(ref_dirs),
        "nsga3": NSGA3(ref_dirs)
    }.get(algorithm_type)
    
    if not algorithm:
        raise ValueError("Invalid algorithm type")
    
    res = minimize(problem, algorithm, termination=('n_gen', 100), verbose=False)
    best_solution = res.X[np.argmin(res.F[:, 1])]
    total = np.sum(best_solution)
    normalized_allocation = {stock: weight / total for stock, weight in zip(future_prices.keys(), best_solution)}
    mean_risk = np.mean(res.F[:, 1]) if np.mean(res.F[:, 1]) != 0 else 1
    confidence = (1 - np.std(res.F[:, 1]) / mean_risk) * 100
    
    return {"allocation": normalized_allocation, "confidence": confidence}

@app.post("/optimize-portfolio")
def get_optimized_portfolio(request: TickerRequest):
    data = fetch_historical_data(request.tickers)
    models = train_ann(data)
    future_prices = predict_future_prices(models, data)
    
    results = {
        "nsga2": optimize_portfolio(future_prices, "nsga2"),
        "moead": optimize_portfolio(future_prices, "moead"),
        "nsga3": optimize_portfolio(future_prices, "nsga3"),
        "FuturePrices": {ticker: list(map(float, prices)) for ticker, prices in future_prices.items()} 
    }
    
    return JSONResponse(content=results, status_code=200)



class SectorRequest(BaseModel):
    sector: str

def fetch_sector_data(sector: str):
    try:
        sector_data = yf.Sector(sector)
        industries_df = sector_data.industries
        
        industries = {row["name"]: row["symbol"] for _, row in industries_df.iterrows()}
        
        return {
            "ticker": sector_data.ticker,
            "top_companies": sector_data.top_companies,
            "top_etfs": sector_data.top_etfs,
            "top_mutual_funds": sector_data.top_mutual_funds,
            "research_reports": sector_data.research_reports,
            "industries": industries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sector data: {str(e)}")

def optimize_industries(industries: Dict[str, str]) -> List[str]:
    future_prices = {}
    
    for industry, symbol in industries.items():
        try:
            ticker = yf.Ticker(symbol)
            stock_prices = ticker.history(period="5y")["Close"].dropna().values
            if stock_prices.size > 0 and not np.isnan(stock_prices).all():
                future_prices[industry] = np.mean(stock_prices)
        except Exception:
            pass

    if not future_prices:
        raise HTTPException(status_code=500, detail="No valid stock data available for optimization.")

    # Multi-objective optimization setup
    problem = PortfolioOptimization(future_prices)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    algorithm = NSGA3(ref_dirs)
    
    res = minimize(problem, algorithm, termination=("n_gen", 100), verbose=False)
    top_indices = np.argsort(res.F[:, 1])[:5]
    
    return [list(future_prices.keys())[i] for i in top_indices]

def get_top_companies(industry_symbol: str) -> List[str]:
    try:
        if not industry_symbol or "^" in industry_symbol:
            return []  # Skip invalid symbols
        
        industry_data = yf.Industry(industry_symbol)
        return industry_data.top_companies["symbol"].tolist()[:5] if industry_data.top_companies is not None else []
    
    except Exception:
        return []

@app.post("/top-industries")
def get_top_industries(request: SectorRequest):
    try:
        sector_data = fetch_sector_data(request.sector)
        return {"industry": sector_data["top_companies"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/historical-data")
def get_historical_data(request: TickerRequest):
    try:
        data = {}
        for ticker in request.tickers:
            stock = yf.Ticker(ticker)
            history = stock.history(period="5y")
            data[ticker] = [{"date": str(date.date()), "close": row["Close"]} for date, row in history.iterrows()]
            print(data)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/search")
def search_stocks(q: str = Query(..., min_length=1)):
    try:
        results = search(q)
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
