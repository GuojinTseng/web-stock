import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

POPULAR_STOCKS = [
    "sh600519", "sz000858", "sh600036", "sh601318", "sz000333", "sz002594", 
    "sh600276", "sh601888", "sz000001", "sh601012", "sz000651", "sh600030", 
    "sh601166", "sh601328", "sh601288", "sh601988", "sh600900", "sz002415", 
    "sh600887", "sz000002", "sh600104", "sh601628", "sz002714", "sh600438", 
    "sh600009", "sh600690", "sz002304", "sh601899", "sh603259", "sz002475",
    "sh601088", "sz000568", "sh600048", "sh601668", "sz002142", "sh601398",
    "sz000157", "sh601816", "sh601319", "sz002493"
]

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/market")
async def get_market_data():
    try:
        url = f"https://qt.gtimg.cn/q={','.join(POPULAR_STOCKS)}"
        res = requests.get(url, timeout=10)
        data = []
        for line in res.text.strip().split('\n'):
            if not line: continue
            parts = line.split('=')[1].strip('"').split('~')
            if len(parts) > 32:
                name = parts[1]
                full_code = line.split('=')[0].split('_')[1]
                price = float(parts[3])
                change_pct = float(parts[32])
                data.append({
                    "代码": full_code,
                    "名称": name,
                    "最新价": price,
                    "涨跌幅": change_pct
                })
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/stock/{symbol}/intraday")
async def get_intraday_data(symbol: str):
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={symbol}"
        res = requests.get(url, timeout=10)
        json_data = res.json()
        
        if json_data['code'] != 0:
            return {"status": "error", "message": "No data"}
            
        points_data = json_data['data'][symbol]['data']['data']
        prices = [float(p.split(' ')[1]) for p in points_data]
        
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
