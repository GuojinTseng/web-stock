"""
青果网络代理 IP 测试脚本
参考: https://www.qg.net/doc/1697.html
"""

import requests

# 账密模式配置（生产环境建议使用环境变量）
AUTH_KEY = "OIUHV2S7"
AUTH_PWD = "E2715439CDD1"

# 提取代理 IP 的 API（弹性提取模式）
# 若为其他模式，请参考: https://www.qg.net/doc/2145.html
EXTRACT_API = "https://share.proxy.qg.net/get"


def get_proxy():
    """从青果 API 提取代理 IP，返回 (代理配置, 出口IP)"""
    params = {"key": AUTH_KEY, "num": 1}
    resp = requests.get(EXTRACT_API, params=params, timeout=10)
    data = resp.json()

    if data.get("code") != "SUCCESS":
        raise RuntimeError(f"提取代理失败: {data}")

    items = data.get("data", [])
    if not items:
        raise RuntimeError("未获取到代理 IP，请检查余额或配额")

    item = items[0]
    server = item.get("server")  # 格式: "IP:端口"
    proxy_ip = item.get("proxy_ip")  # 真实出口 IP
    if not server:
        raise RuntimeError("响应中无 server 字段")

    proxy_url = "http://%(user)s:%(password)s@%(server)s" % {
        "user": AUTH_KEY,
        "password": AUTH_PWD,
        "server": server,
    }
    proxies = {"http": proxy_url, "https": proxy_url}
    return proxies, proxy_ip


def test_proxy():
    """使用代理访问测试 URL"""
    target_url = "https://www.baidu.com"

    print("正在提取代理 IP...")
    proxies, proxy_ip = get_proxy()
    print(f"当前使用的 IP: {proxy_ip}")
    print("代理已获取，正在测试...")

    resp = requests.get(target_url, proxies=proxies, timeout=15)
    print(f"状态码: {resp.status_code}")
    print(f"响应内容:\n{resp.text}")


if __name__ == "__main__":
    test_proxy()
