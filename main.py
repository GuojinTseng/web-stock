import os
import requests
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import akshare as ak

# For the AI Analysis
try:
    import deepseek_client
except ImportError:
    deepseek_client = None

# For IP Proxy Pool
try:
    from ip import get_proxy
except ImportError:
    get_proxy = None


app = FastAPI(title="Web Stock Data")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply proxy for testing if needed
# Note: For EastMoney requests in production you'd use get_proxy() and inject via os.environ HTTP_PROXY
def apply_proxy_if_available():
    if get_proxy:
        try:
            proxies, proxy_ip = get_proxy()
            proxy_url = proxies.get("https") or proxies.get("http")
            if proxy_url:
                os.environ["HTTP_PROXY"] = proxy_url
                os.environ["HTTPS_PROXY"] = proxy_url
                # Clean up any NO_PROXY that might interfere
                if "NO_PROXY" in os.environ:
                    del os.environ["NO_PROXY"]
                print(f"[Proxy Configured] IP: {proxy_ip}")
        except Exception as e:
            print(f"[Proxy Error] {e}")

# Call it on startup
apply_proxy_if_available()
# Configure deepseek via environment variable
dp_key = os.environ.get("DEEPSEEK_API_KEY", "")
if deepseek_client and dp_key:
    deepseek_client.configure(dp_key)

# (Removed hardcoded POPULAR_STOCKS)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

import time

def call_akshare_with_retry(func, *args, retries=3, **kwargs):
    """
    Wrap Akshare calls to handle proxy disconnection by refreshing the proxy.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Akshare] Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print("[Akshare] Refreshing proxy and retrying...")
                apply_proxy_if_available()
                time.sleep(1)
            else:
                raise e

@app.get("/api/market")
async def get_market_data(symbols: str = ""):
    try:
        if not symbols:
            symbols = "sh600519,sz000858,sz000001"
            
        dataset = call_akshare_with_retry(ak.stock_zh_a_spot_em)
        data = []
        target_codes = [s.strip() for s in symbols.split(',') if s.strip()]
        target_bare_codes = [s[2:] for s in target_codes if len(s) > 2]
        
        filtered = dataset[dataset['代码'].isin(target_bare_codes)]
        for _, row in filtered.iterrows():
            bare_code = str(row['代码'])
            prefix = "sh" if bare_code.startswith(("6", "9")) else "sz"
            full_code = f"{prefix}{bare_code}"
            
            data.append({
                "代码": full_code,
                "名称": row['名称'],
                "最新价": float(row['最新价']) if not type(row['最新价']) == str else 0.0,
                "涨跌幅": float(row['涨跌幅']) if not type(row['涨跌幅']) == str else 0.0
            })
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/stock/{symbol}/intraday")
async def get_intraday_data(symbol: str):
    try:
        df = call_akshare_with_retry(ak.stock_intraday_em, symbol=symbol)
        if df.empty:
            return {"status": "error", "message": "No data"}
            
        prices = df['成交价'].astype(float).tolist()
        
        # Downsample to ~50 points so it renders fast in SVG
        step = max(1, len(prices) // 50)
        sampled = prices[::step]
        return {"status": "success", "data": sampled}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/stock/{symbol}/analyze")
async def analyze_stock(symbol: str):
    if not deepseek_client or not deepseek_client.is_configured():
        async def mock_stream():
            yield "data: ⚠️ 未配置 DeepSeek API 密钥，AI 诊断暂不可用。\n\n"
        return StreamingResponse(mock_stream(), media_type="text/event-stream")

    try:
        # First grab some very basic context to feed the AI
        url = f"https://qt.gtimg.cn/q={symbol}"
        res = requests.get(url, timeout=5)
        text_data = res.text
        
        # We craft a quick real-time prompt
        messages = deepseek_client.build_realtime_analysis_prompt(symbol, text_data, "")

        async def stream_generator():
            # The deepseek_client.chat_stream is synchronous, so we run it in a thread or just iterate
            # Since fastAPI async routes block thread if we do blocking I/O, we should use threadpool
            def sync_stream():
                for chunk in deepseek_client.chat_stream(messages, max_tokens=1500):
                    yield chunk

            # A very simple wrapper to make sync generator async compatible enough for StreamingResponse
            # To do this robustly in prod, using anyio to unblock is best
            loop = asyncio.get_event_loop()
            iterator = iter(sync_stream())
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, iterator)
                    # Format for SSE
                    line = chunk.replace('\n', '\\n')
                    yield f"data: {chunk}\n\n"
                except StopIteration:
                    break
                except Exception as e:
                    yield f"data: [error: {e}]\n\n"
                    break

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as e:
        async def err_stream():
            yield f"data: ⚠️ 生成报告时出错：{e}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    apply_proxy_if_available()
    # Zeabur 部署时会自动注入 PORT 环境变量
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
