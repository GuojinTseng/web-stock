import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import akshare as ak
import uvicorn

app = FastAPI(title="Web Stock Data")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/market")
async def get_market_data():
    try:
        # 获取最新 A 股实时行情数据（东方财富）
        dataset = ak.stock_zh_a_spot_em()
        
        # 为了演示响应速度，只取前 40 支股票
        top_stocks = dataset.head(40).to_dict(orient="records")
        return {"status": "success", "data": top_stocks}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/stock/{symbol}/intraday")
async def get_intraday_data(symbol: str):
    try:
        df = ak.stock_intraday_em(symbol=symbol)
        if df.empty:
            return {"status": "error", "message": "No data"}
        
        # We only need '成交价' for a sparkline
        prices = df['成交价'].astype(float).tolist()
        
        # Downsample to ~50 points so it renders fast in SVG
        step = max(1, len(prices) // 50)
        sampled = prices[::step]
        return {"status": "success", "data": sampled}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Zeabur 部署时会自动注入 PORT 环境变量
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
