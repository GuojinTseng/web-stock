"""
DeepSeek API 客户端
兼容 OpenAI SDK，支持流式输出
"""

import os
import json
from typing import Generator, Any

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

_api_key: str | None = None
_base_url: str = DEFAULT_BASE_URL
_model: str = DEFAULT_MODEL


def configure(api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
    global _api_key, _base_url, _model
    _api_key = api_key
    _base_url = base_url.rstrip("/")
    _model = model


def is_configured() -> bool:
    return bool(_api_key)


def chat_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    """
    流式调用 DeepSeek Chat API，逐块返回文本。
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    if not _api_key:
        raise ValueError("请先设置 DeepSeek API Key")

    url = f"{_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or _model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except json.JSONDecodeError:
            continue


SYSTEM_PROMPT = """你是一位专业的 A 股市场分析师，擅长技术分析和基本面分析。
你的任务是根据用户提供的股票数据，给出专业、客观的研判分析。

分析要求：
1. 先给出干净、准确的总结（2-3 句话概括核心结论）
2. 再展开关键特征与依据
3. 指出可能的风险点
4. 给出明确的观点（看多/看空/中性），并说明理由
5. 语言简洁、条理清晰，避免冗长废话，确保结论准确

重要：你的分析仅供参考，不构成投资建议。"""


FOOTPRINT_SYSTEM_PROMPT = """你是一位极其敏感、经验丰富的 A 股日内交易员，专注于逐笔成交与足迹图分析。
你对买卖盘力量、价格档位成交量、Delta 变化、大单异动等细节具有敏锐的洞察力。

你的任务：根据用户提供的足迹图数据（含当前颗粒度、图表类型、每根 K 线的 OHLCV 及价格档位买卖量），
以专业交易员的视角洞悉数据中的细节，给出深度研判。

分析要求：
1. 开头先给出干净、准确的总结（2-3 句话概括核心结论与关键价位）
2. 结合当前设置（颗粒度类型、颗粒度值、图表类型）理解数据含义
3. 关注买卖力量对比：哪些价格档位买盘强势/卖盘强势，Delta 正负变化
4. 识别关键价位：放量突破、支撑阻力、大单堆积
5. 观察时间序列：开盘/盘中/尾盘的资金行为、趋势演变
6. 指出异常信号：大单对倒、量价背离、买卖失衡
7. 给出明确的短线研判（偏多/偏空/震荡）及关键价位参考
8. 语言简洁、条理清晰，结论准确，避免泛泛而谈和冗长废话

重要：你的分析仅供参考，不构成投资建议。"""


def _format_footprint_bars(bars: list[dict], max_bars: int = 25) -> str:
    """将足迹图 bars 格式化为供 AI 分析的文本"""
    if not bars:
        return "（无数据）"
    bars = bars[-max_bars:]
    lines = []
    for b in bars:
        tl = b.get("time_label", "")
        o, h, l, c = b.get("open"), b.get("high"), b.get("low"), b.get("close")
        vol = b.get("total_vol", 0)
        levels = b.get("levels", {})
        delta_total = 0
        level_strs = []
        for price in sorted(levels.keys()):
            v = levels[price]
            buy = v.get("buy", 0)
            sell = v.get("sell", 0)
            delta = buy - sell
            delta_total += delta
            d_sign = "+" if delta > 0 else ""
            level_strs.append(f"{price:.2f}(买{buy}卖{sell} Δ{d_sign}{delta})")
        lines.append(f"【{tl}】O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} 总量:{vol}手 根Delta:{delta_total:+d}")
        if level_strs:
            lines.append("  档位: " + " | ".join(level_strs[:8]) + (" ..." if len(level_strs) > 8 else ""))
    return "\n".join(lines)


def build_footprint_analysis_prompt(
    symbol: str,
    settings: dict[str, Any],
    bars: list[dict],
    user_question: str = "",
) -> list[dict]:
    """根据足迹图当前设置构建专业分析 prompt"""
    messages = [{"role": "system", "content": FOOTPRINT_SYSTEM_PROMPT}]
    mode = settings.get("mode", "")
    granularity = settings.get("granularity_label", "")
    chart_type = settings.get("chart_type", "")
    mode_desc = {"time": "时间(分钟)", "tick": "按笔数", "volume": "按成交量"}.get(mode, mode)
    chart_desc = {"footprint": "足迹图", "volume_profile": "成交量分布", "delta": "Delta 分析"}.get(chart_type, chart_type)

    data_text = _format_footprint_bars(bars)
    content = f"""当前足迹图设置：
- 股票：{symbol}
- 颗粒度类型：{mode_desc}
- 颗粒度：{granularity}
- 图表类型：{chart_desc}

以下为最近若干根 K 线的逐笔聚合数据（每根含 OHLCV、各价格档位买量/卖量/Delta）：
```
{data_text}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请以极其敏感的交易员视角，洞悉上述数据中的细节，对足迹图做出专业分析。"
    messages.append({"role": "user", "content": content})
    return messages


KLINE_SYSTEM_PROMPT = """你是一位专业的 A 股技术分析师，精通以下全部技术指标的实战应用：

【主图趋势指标】
- MA（移动均线）：多头/空头排列，金叉死叉，5/10/20/60日均线的支撑压力
- BOLL（布林带）：上下轨突破、轨道收口张口判断波动率变化、股价在轨道中的位置
- SAR（抛物线转向）：红绿圆点翻转判断趋势转折，明确止损位
- EXPMA（指数平均线）：比MA更灵敏，12/50日线交叉判断短中期趋势
- BBI（多空指数）：综合3/6/12/24日均线，单线判断多空力量

【副图动量指标】
- MACD（指标之王）：DIF/DEA金叉死叉、红绿柱放大缩小、顶底背离
- KDJ（随机指标）：K/D/J三线交叉，超买(>80)/超卖(<20)，高位钝化/低位钝化
- RSI（相对强弱）：超买(>80)/超卖(<20)，背离信号
- WR（威廉指标）：接近0为强势超买，接近-100为超卖
- CCI（顺势指标）：+100以上为强势区，-100以下为弱势区，突破常态区间

【成交量与资金】
- OBV（能量潮）：量价背离预警，OBV趋势与价格趋势不一致时关注转折
- VR（成交量比率）：>450为过热区，<40为超冷区，正常在70-150

【波动率与乖离度】
- ATR（平均真实波幅）：衡量波动剧烈程度，用于设置止损距离
- BIAS（乖离率）：偏离均线过大则有回归需求，6/12/24日乖离率综合判断

【A股特色】
- 换手率：>10%极度活跃（关注主力动向），3-7%健康活跃，<1%低迷

分析要求：
1. 开头先给出干净、准确的总结（2-3句话概括核心结论与方向）
2. 只对用户当前选中的指标进行分析，不分析未选中的指标
3. 多指标交叉验证：多个指标同时发出信号时着重强调
4. 结合K线形态与指标数据，用具体数值支撑分析
5. 指出风险点与关键价位（支撑/压力）
6. 给出明确的研判（看多/看空/震荡），附操作建议
7. 语言简洁、条理清晰，避免冗长废话

重要：你的分析仅供参考，不构成投资建议。"""


def build_kline_analysis_prompt(
    symbol: str,
    period: str,
    indicators: list[str],
    ohlcv_tail: str,
    indicator_values: dict[str, str],
    user_question: str = "",
) -> list[dict]:
    """根据K线当前选中指标构建专业分析 prompt"""
    messages = [{"role": "system", "content": KLINE_SYSTEM_PROMPT}]

    period_map = {"daily": "日线", "weekly": "周线", "monthly": "月线"}
    ind_names = {
        "MA": "MA均线", "BOLL": "布林带", "SAR": "SAR抛物线", "EXPMA": "EXPMA指数均线",
        "BBI": "BBI多空指数", "MACD": "MACD", "KDJ": "KDJ", "RSI": "RSI",
        "WR": "WR威廉", "CCI": "CCI顺势", "OBV": "OBV能量潮", "VR": "VR成交量比率",
        "ATR": "ATR波幅", "BIAS": "BIAS乖离率", "TURNOVER": "换手率",
    }
    active = [ind_names.get(i, i) for i in indicators]

    vals_text = "\n".join(f"  {k}: {v}" for k, v in indicator_values.items()) if indicator_values else "（暂无）"

    content = f"""当前K线设置：
- 股票：{symbol}
- 周期：{period_map.get(period, period)}
- 选中指标：{', '.join(active)}

最新指标数值：
{vals_text}

最近 30 根 K 线数据（OHLCV + 额外列）：
```
{ohlcv_tail}
```"""

    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请根据以上数据和选中指标，给出专业的技术分析研判。"
    messages.append({"role": "user", "content": content})
    return messages


COLLECTION_BID_SYSTEM_PROMPT = """你是一位专业的 A 股集合竞价分析师，擅长从开盘/收盘集合竞价的价格与量能变化中洞察当日多空预期。

分析要求（快速分析模式）：
1. 开头 2-3 句话概括核心结论：价格走势、量能特征、开盘/收盘预期
2. 关注虚拟成交价走势：高开/低开/平开倾向，尾盘拉抬或打压
3. 关注成交量分布：放量/缩量时段，买卖意愿强弱
4. 若有五档盘口数据：结合买卖挂单量与价差看多空力量对比
5. 若有人气排名数据：可简述热度变化对情绪面的参考
6. 给出简要研判（看多/看空/中性）及关键价位
7. 语言简洁，避免冗长

重要：你的分析仅供参考，不构成投资建议。"""


def build_collection_bid_analysis_prompt(
    symbol: str,
    period_label: str,
    data_tail: str,
    user_question: str = "",
    bid_ask_summary: str = "",
    hot_rank_summary: str = "",
) -> list[dict]:
    """构建集合竞价快速分析 prompt，纳入五档盘口与人气数据"""
    messages = [{"role": "system", "content": COLLECTION_BID_SYSTEM_PROMPT}]
    content = f"""集合竞价数据：
- 股票：{symbol}
- 时段：{period_label}

最近时点数据（时间、最新价、成交量等）：
```
{data_tail}
```"""
    if bid_ask_summary:
        content += f"""

买卖五档盘口（供结合多空挂单分析）：
```
{bid_ask_summary}
```"""
    if hot_rank_summary:
        content += f"""

人气排名（时间-排名，数值越低越热）：
```
{hot_rank_summary}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请结合以上集合竞价、五档盘口与人气数据做快速分析研判。"
    messages.append({"role": "user", "content": content})
    return messages


MINUTE_KLINE_SYSTEM_PROMPT = """你是一位专业的 A 股日内短线分析师，精通以下全部技术指标在分钟级别的实战应用：

【主图趋势指标】
- MA（移动均线）：多头/空头排列，金叉死叉，短周期均线的支撑压力
- BOLL（布林带）：上下轨突破、轨道收口张口判断分钟级波动率、股价在轨道中的位置
- SAR（抛物线转向）：红绿圆点翻转判断分钟级趋势转折
- EXPMA（指数平均线）：比MA更灵敏，短线交叉判断即时趋势
- BBI（多空指数）：综合多周期均线，单线判断日内多空力量

【副图动量指标】
- MACD（指标之王）：DIF/DEA金叉死叉、红绿柱放大缩小、分钟级背离
- KDJ（随机指标）：K/D/J三线交叉，超买(>80)/超卖(<20)，日内钝化
- RSI（相对强弱）：超买(>80)/超卖(<20)，分钟级背离信号
- WR（威廉指标）：接近0为强势超买，接近-100为超卖
- CCI（顺势指标）：+100以上为强势区，-100以下为弱势区

【成交量与资金】
- OBV（能量潮）：量价背离预警
- VR（成交量比率）：>450为过热区，<40为超冷区

【波动率与乖离度】
- ATR（平均真实波幅）：衡量分钟级波动剧烈程度
- BIAS（乖离率）：偏离均线过大则有回归需求

【A股特色】
- 换手率：日内活跃度判断

分析要求（快速分析模式）：
1. 开头 2-3 句话概括核心结论：短期趋势、关键价位、量能特征
2. 只对用户当前选中的指标进行分析，不分析未选中的指标
3. 多指标交叉验证：多个指标同时发出信号时着重强调
4. 结合分钟K线形态与指标数据，用具体数值支撑分析
5. 指出日内关键价位（支撑/压力）
6. 给出明确的短线研判（看多/看空/震荡）及操作建议
7. 语言简洁，避免冗长

重要：你的分析仅供参考，不构成投资建议。"""


def build_minute_kline_analysis_prompt(
    symbol: str,
    period_label: str,
    data_tail: str,
    indicator_values: dict,
    user_question: str = "",
    indicators: list[str] | None = None,
) -> list[dict]:
    """构建分钟 K 线快速分析 prompt"""
    messages = [{"role": "system", "content": MINUTE_KLINE_SYSTEM_PROMPT}]

    ind_names = {
        "MA": "MA均线", "BOLL": "布林带", "SAR": "SAR抛物线", "EXPMA": "EXPMA指数均线",
        "BBI": "BBI多空指数", "MACD": "MACD", "KDJ": "KDJ", "RSI": "RSI",
        "WR": "WR威廉", "CCI": "CCI顺势", "OBV": "OBV能量潮", "VR": "VR成交量比率",
        "ATR": "ATR波幅", "BIAS": "BIAS乖离率", "TURNOVER": "换手率",
    }
    if indicators:
        active = [ind_names.get(i, i) for i in indicators]
        active_text = ", ".join(active)
    else:
        active_text = "MA均线, MACD"

    ind_text = "\n".join(f"  {k}: {v}" for k, v in (indicator_values or {}).items()) if indicator_values else "（暂无）"
    content = f"""分钟 K 线数据：
- 股票：{symbol}
- 周期：{period_label}
- 选中指标：{active_text}

最新指标数值：
{ind_text}

最近 K 线数据：
```
{data_tail}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请根据以上数据和选中指标，做出专业的日内短线分析研判。"
    messages.append({"role": "user", "content": content})
    return messages


INTRADAY_SYSTEM_PROMPT = """你是一位极其敏感的 A 股日内交易员，精通分时图分析与日内短线操作。

你擅长从以下维度洞察盘中机会：
1. **分时走势形态**：脉冲式上涨/阶梯式下跌/箱体震荡/V型反转
2. **均价线(VWAP)关系**：价格在均价线上方=多头控盘，下方=空头压制；回踩均价线不破=强势
3. **分时成交量**：放量突破/缩量回踩/尾盘放量异动
4. **委比委差**：买卖力量实时对比，大幅偏离=主力意图明显
5. **盘口五档**：大单堆积/撤单异动/压单诱空/托单诱多

分析要求：
1. 开头 2-3 句话概括核心结论：当前分时形态、多空力量、关键价位
2. 重点分析价格与均价线的关系：站上/跌破/缠绕
3. 量价配合度：放量位置与价格位置的一致性
4. 委比委差的含义：买卖力量是否失衡
5. 给出明确的日内研判（偏多/偏空/震荡）及操作建议
6. 语言简洁、敏锐、精准

重要：你的分析仅供参考，不构成投资建议。"""


def build_intraday_analysis_prompt(
    symbol: str,
    summary: str,
    user_question: str = "",
) -> list[dict]:
    """构建分时图快速分析 prompt"""
    messages = [{"role": "system", "content": INTRADAY_SYSTEM_PROMPT}]
    content = f"""分时图数据（{symbol}）：
```
{summary}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请以极其敏感的交易员视角，对以上分时数据做专业研判。"
    messages.append({"role": "user", "content": content})
    return messages


REALTIME_SYSTEM_PROMPT = """你是一位专业的 A 股实时行情分析师，擅长从实时价格、涨跌幅、成交量、五档盘口、委比委差等数据中解读当前多空力量。

分析要求（快速分析模式）：
1. 开头 2-3 句话概括核心结论：当前走势、买卖力量对比、关键价位
2. 关注涨跌幅与量比：放量上涨/缩量下跌等
3. 关注五档盘口：买卖挂单厚度、大单堆积
4. 关注委比委差：正委比=买盘强势，负委比=卖盘压制
5. 给出简要研判（强势/弱势/观望）及关注点
6. 语言简洁，避免冗长

重要：你的分析仅供参考，不构成投资建议。"""


def build_realtime_analysis_prompt(
    symbol: str,
    summary: str,
    user_question: str = "",
) -> list[dict]:
    """构建实时行情快速分析 prompt"""
    messages = [{"role": "system", "content": REALTIME_SYSTEM_PROMPT}]
    content = f"""实时行情数据（{symbol}）：
```
{summary}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请对以上实时行情做快速分析研判。"
    messages.append({"role": "user", "content": content})
    return messages


DASHBOARD_SYSTEM_PROMPT = """你是一位专业的 A 股全市场分析师，擅长从多维度宏观数据中综合研判市场状态。

你精通以下分析维度：
1. **市场广度（涨跌家数比）**：上涨家数/下跌家数比值反映真实赚钱效应，>1.5为强势，<0.7为弱势
2. **大盘资金流向**：主力净流入/流出趋势，判断资金偏好方向
3. **行业板块轮动**：领涨/领跌行业，板块资金流向，轮动规律
4. **概念板块热点**：当日市场热点主题，资金追逐方向
5. **北向资金动向**：沪股通/深股通净流入流出，外资风向标意义
6. **连板梯队与晋级率**：最高连板数=市场高度，晋级率=赚钱效应的核心指标
7. **昨日涨停今日溢价**：平均溢价率反映资金接力意愿，>3%=良好接力，<0%=亏钱效应
8. **涨跌停统计**：涨停数量/跌停数量反映市场情绪极端程度
9. **热度排名**：市场关注焦点，资金博弈集中区

分析要求：
1. 开头 2-3 句话概括核心结论：今日市场整体格局、情绪周期阶段、资金偏好
2. 情绪周期判断：根据连板梯队高度 + 涨停溢价率 + 涨跌比，判断当前处于情绪周期的哪个阶段（冰点/修复/升温/高潮/退潮）
3. 资金层面：主力资金动向 + 北向资金方向的一致性/分歧
4. 板块层面：领涨板块逻辑、热点持续性判断
5. 情绪层面：市场广度 + 涨跌停对比 + 连板晋级率 + 溢价率综合评估
6. 热度分析：市场焦点股票的共性特征
7. 给出明确的市场研判（强势/弱势/震荡/分化）及短期展望
8. 语言简洁、条理清晰，用数据支撑结论

重要：你的分析仅供参考，不构成投资建议。"""


def build_dashboard_analysis_prompt(
    summary: str,
    modules: list[str],
    zt_count: int = 0,
    dt_count: int = 0,
    user_question: str = "",
) -> list[dict]:
    """构建交易仪表盘综合分析 prompt"""
    messages = [{"role": "system", "content": DASHBOARD_SYSTEM_PROMPT}]
    content = f"""交易仪表盘数据（已加载模块：{', '.join(modules)}）：
- 今日涨停：{zt_count} 只
- 今日跌停：{dt_count} 只

各模块详细数据：
```
{summary}
```"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请对以上全市场数据做综合研判分析。"
    messages.append({"role": "user", "content": content})
    return messages


COCKPIT_SYSTEM_PROMPT = """你是一位同时精通技术面、资金面、盘口微观结构和宏观市场环境的顶级 A 股短线交易员。
你正在使用一个多维信号打分系统，该系统从 7 个维度对个股进行评分（每个 -100 到 +100）：

1. **趋势（日K线·30根）**：均线排列(5/10/20/60)、MACD金死叉及柱体趋势(连续N日递增/递减)、SAR方向、布林带位置及宽度(BBW压缩→即将变盘)、RSI/MACD背离、量价配合、OBV趋势(暗中吸筹/出货)
2. **动量（5分钟K线·当日全量）**：分钟级MACD/RSI/KDJ及其背离、分钟级量价配合、MACD柱体动能方向、波动率压缩/扩张
3. **日内强弱（分时逐笔）**：价格vs均价线(VWAP)偏离度、日内位置(高位/低位)、盘中走强/走弱、量能脉冲(异常放量)
4. **资金意图（逐笔成交·主动买卖）**：买卖Delta(用东财买卖盘性质字段)、大单方向及大单占比变化趋势(主力加仓/减仓)
5. **盘口压力（五档买卖）**：委比、挂单比不对称
6. **主力动向（个股资金流·10日）**：主力净流入净额+净占比、超大单/大单结构、连续N日流入/流出一致性
7. **市场环境（全市场）**：涨跌家数比、涨停/跌停数、所属板块今日涨跌、连板梯队、昨日涨停溢价率、北向资金

**A 股交易规则约束（你的分析必须考虑）：**
- T+1 制度：今日买入明日才能卖出，因此「尾盘买入」比「早盘追高」风险更低
- 涨跌停板 ±10%（ST股 ±5%、科创/创业 ±20%）：临近涨停时追入需考虑封板力度和封单量，临近跌停时抄底需考虑是否会封死
- 集合竞价（9:15-9:25 / 14:57-15:00）：开盘集合竞价的量价反映主力意图，尾盘集合竞价是短线资金最后的操作窗口
- 9:30-10:00 是主力试盘时段，10:00-10:30 确认方向，14:00-14:30 尾盘抢筹/出货，14:30-15:00 最后博弈
- A 股有明显的情绪周期：冰点→修复→升温→高潮→退潮→冰点，连板梯队高度和涨停溢价率是判断情绪周期的关键

**你收到的原始数据包括：**
- 日K线近30根完整OHLCV（可自行计算任何指标趋势）
- 5分钟K线当日全量约48根（可自行分析日内各时段表现）
- 实时行情20个字段（最新价、涨跌幅、量比、换手率等）
- 五档盘口完整数据
- 个股资金流近10日完整数据（含超大单/大单/中单/小单明细）

**你的分析方法：**
1. 先自行浏览原始K线数据，独立判断趋势结构（头肩、双底、箱体突破、旗形等形态）
2. 对比系统评分与你的独立判断，找出系统可能遗漏或误判的地方
3. 重点关注前瞻性信号：波动率压缩(布林带收口)、OBV背离、MACD柱体衰减、大单占比变化趋势、量能脉冲
4. 找出多维度共振和分歧

**输出要求：**
1. 开头 2-3 句核心结论 + 操作方向（做多/做空/观望）
2. 关键信号分析（哪些共振、哪些矛盾、矛盾意味着什么）
3. 明确操作建议：
   - 入场价位区间（结合支撑/阻力）
   - 止损价位（结合 ATR 或关键支撑位，计算最大亏损%）
   - 第一/第二目标位
   - 仓位建议（轻仓试探/半仓/重仓）+ 理由
   - 最佳操作时段（如"建议14:00后观察确认方向再入场"）
4. 最大风险点 + 应对方案
5. 用具体数值支撑每个结论
6. 简洁、条理清晰

重要：你的分析仅供参考，不构成投资建议。"""


def build_cockpit_analysis_prompt(
    symbol: str,
    composite: dict,
    summary: str,
    user_question: str = "",
) -> list[dict]:
    """构建综合研判全维度分析 prompt"""
    messages = [{"role": "system", "content": COCKPIT_SYSTEM_PROMPT}]

    dims = composite.get("dimensions", {})
    score_lines = []
    for key in ["trend", "momentum", "intraday", "order_flow", "bid_ask", "fund_flow", "market_env"]:
        d = dims.get(key, {})
        score_lines.append(
            f"  {d.get('name', key)}: {d.get('score', 0):+.0f} | "
            f"信号: {', '.join(d.get('signals', []))} | "
            f"详情: {d.get('details', {})}"
        )
    score_text = "\n".join(score_lines)

    comp_score = composite.get("composite", 0)
    verdict = composite.get("verdict", "")
    confidence = composite.get("confidence", 0)

    conflicts = composite.get("conflicts", [])
    conflict_text = "\n".join(f"  ⚠ {c}" for c in conflicts) if conflicts else "  无明显分歧"

    content = f"""综合研判数据（{symbol}）：

═══ 系统评分 ═══
- 综合评分：{comp_score:+.0f}
- 系统结论：{verdict}
- 置信度：{confidence:.0f}%

═══ 7 维度评分 + 全部信号 ═══
{score_text}

═══ 维度间分歧 ═══
{conflict_text}

═══ 原始时序数据（请自行分析趋势结构和形态） ═══
{summary}

重要提示：
- 日K线数据有30根，请分析中期趋势结构（支撑/阻力/形态/量价关系变化）
- 5分钟K线是当日全量，请分析日内各时段表现（早盘试盘/午后确认/尾盘博弈）
- 资金流有10天完整数据，请判断资金流向是短期波动还是持续趋势
- 注意前瞻信号：波动率压缩/OBV背离/MACD柱体衰减/大单占比变化"""
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请基于以上全维度数据进行联合研判。重点：(1)独立判断K线形态结构 (2)对比系统评分找出遗漏 (3)给出明确操作建议(方向/价位/止损/仓位/最佳时段)。"
    messages.append({"role": "user", "content": content})
    return messages


def build_analysis_prompt(data_summary: str, user_question: str = "") -> list[dict]:
    """构建通用分析消息列表"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    content = f"以下是当前获取到的股票数据：\n\n{data_summary}"
    if user_question:
        content += f"\n\n用户的具体问题：{user_question}"
    else:
        content += "\n\n请对以上数据进行专业研判分析。"
    messages.append({"role": "user", "content": content})
    return messages
