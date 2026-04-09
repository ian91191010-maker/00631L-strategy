import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 網頁 UI 設定
# ==========================================
st.set_page_config(page_title="動態量化交易策略 (自適應升級版)", layout="wide")
st.title("00631L.TW 動態波動率自適應策略模型")
st.markdown("本系統基於 Yahoo Finance 歷史數據運算。已修復原策略之參數過度配適、槓桿波動耗損及進出場不對稱缺陷。")

# 側邊欄參數設定區
st.sidebar.header("動態模型參數控制面板")
ticker = st.sidebar.text_input("分析標的 (Yahoo Finance 格式)", "00631L.TW")
lookback_years = st.sidebar.slider("回測區間 (年)", 1, 5, 3)

st.sidebar.subheader("波動率與停損參數")
atr_multiplier = st.sidebar.slider("ATR 移動停利乘數", 1.5, 4.0, 2.5, step=0.1)
hv_threshold = st.sidebar.slider("HV20 波動率濾網上限 (%)", 20, 50, 30)
z_score_threshold = st.sidebar.slider("動態超賣 Z-Score 閾值", -3.0, -1.0, -2.0, step=0.1)

# ==========================================
# 核心運算模組
# ==========================================
@st.cache_data
def fetch_data(symbol, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    df = yf.download(symbol, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def calculate_vhf(df, period=35):
    hcp = df['High'].rolling(period).max()
    lcp = df['Low'].rolling(period).min()
    diff_sum = df['Close'].diff().abs().rolling(period).sum()
    return (hcp - lcp) / diff_sum

def run_dynamic_strategy(df):
    df = df.copy()
    
    # 1. 基礎技術指標
    df['SAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
    df.loc[df['SAR'].isna(), 'SAR'] = ta.trend.psar_up(df['High'], df['Low'], df['Close'])
    df['VHF35'] = calculate_vhf(df, period=35).fillna(0)
    
    # 2. 動態超賣指標 (Rolling Z-Score of CCI)
    df['CCI40'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=40)
    cci_mean = df['CCI40'].rolling(252).mean()
    cci_std = df['CCI40'].rolling(252).std()
    df['CCI_Z'] = (df['CCI40'] - cci_mean) / cci_std
    
    # 3. 波動率濾網與停損指標 (HV20 & ATR)
    df['HV20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['ATR14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # 4. 狀態變數平移
    df['Close_prev'] = df['Close'].shift(1)
    df['SAR_prev'] = df['SAR'].shift(1)
    
    # 5. 進場邏輯 (動態超賣 或 趨勢翻多，且通過波動率與趨勢濾網)
    cond_oversold = df['CCI_Z'] < z_score_threshold
    cond_trend_up = (df['Close'] > df['SAR']) & (df['Close_prev'] <= df['SAR_prev'])
    cond_trend_filter = df['VHF35'] > 0.2
    cond_volatility_safe = df['HV20'] < hv_threshold
    
    df['Entry_Signal'] = (cond_oversold | cond_trend_up) & cond_trend_filter & cond_volatility_safe
    
    # 6. 出場邏輯與部位模擬
    df['Position'] = 0
    df['Action'] = ""
    
    in_position = False
    entry_price = 0
    highest_since_entry = 0
    bars_held = 0
    
    for i in range(1, len(df)):
        if not in_position and df['Entry_Signal'].iloc[i]:
            in_position = True
            entry_price = df['Close'].iloc[i]
            highest_since_entry = entry_price
            bars_held = 0
            df.iat[i, df.columns.get_loc('Action')] = "BUY"
            
        elif in_position:
            bars_held += 1
            current_close = df['Close'].iloc[i]
            current_atr = df['ATR14'].iloc[i]
            highest_since_entry = max(highest_since_entry, current_close)
            
            # 出場條件 A: 對稱性趨勢翻空
            exit_trend = current_close < df['SAR'].iloc[i]
            # 出場條件 B: ATR 動態移動停利/停損
            trailing_stop_price = highest_since_entry - (atr_multiplier * current_atr)
            exit_trailing = current_close < trailing_stop_price
            # 出場條件 C: Time Stop (10日不漲則平倉)
            exit_time = (bars_held > 10) and (current_close < entry_price * 1.01)
            
            if exit_trend or exit_trailing or exit_time:
                in_position = False
                df.iat[i, df.columns.get_loc('Action')] = "SELL"
                
        df.iat[i, df.columns.get_loc('Position')] = 1 if in_position else 0
        
    return df

# ==========================================
# 執行與圖表渲染
# ==========================================
if st.sidebar.button("執行模型運算"):
    with st.spinner('正在從 Yahoo Finance 獲取資料並執行運算...'):
        data = fetch_data(ticker, lookback_years)
        if data.empty:
            st.error("無法獲取資料，請確認標的代碼正確。")
        else:
            result_df = run_dynamic_strategy(data)
            
            # 擷取最近100天的資料作圖以保持清晰
            plot_df = result_df.tail(150)
            
            fig = go.Figure()
            
            # K線圖
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                         low=plot_df['Low'], close=plot_df['Close'], name='K線'))
            
            # SAR 指標
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SAR'], mode='markers', 
                                     marker=dict(size=4, color='orange'), name='SAR'))
            
            # 買賣點標記
            buys = plot_df[plot_df['Action'] == 'BUY']
            sells = plot_df[plot_df['Action'] == 'SELL']
            
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers',
                                     marker=dict(symbol='triangle-up', size=12, color='green'), name='買進'))
            fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='red'), name='賣出'))
            
            fig.update_layout(title=f"{ticker} 策略進出場圖表 (近150日交易日)", xaxis_title="日期", yaxis_title="價格", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示近期訊號與數據
            st.subheader("近期交易訊號與指標狀態")
            display_cols = ['Close', 'HV20', 'VHF35', 'CCI_Z', 'ATR14', 'Action', 'Position']
            st.dataframe(result_df[display_cols].tail(10).style.highlight_max(axis=0))