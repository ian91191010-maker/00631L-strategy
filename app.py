import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==========================================
# 網頁 UI 設定 (極簡純淨版)
# ==========================================
st.set_page_config(page_title="正價差與價格行為防禦策略", layout="wide")
st.title("00631L.TW 二維防禦矩陣：正價差環境 + 破底觸發")

st.sidebar.subheader("資料源設定")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")

st.sidebar.subheader("回測與顯示參數")
lookback_years = st.sidebar.number_input("回測年數", min_value=1, max_value=10, value=5)
plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)

# ==========================================
# 資料獲取模組
# ==========================================
@st.cache_data
def fetch_stock_data(symbol, years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date, "end_date": end_date, "token": token}
    try:
        res = requests.get(url, params=parameter, timeout=15)
        df = pd.DataFrame(res.json().get("data", []))
        if df.empty: return df
        df = df.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close']: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Close'])
    except:
        return pd.DataFrame()

@st.cache_data
def fetch_futures_data(years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanFuturesDaily", "data_id": "TX", "start_date": start_date, "end_date": end_date, "token": token}
    try:
        res = requests.get(url, params=parameter, timeout=15)
        df = pd.DataFrame(res.json().get("data", []))
        if df.empty: return df
        # 篩選近月合約並取每日唯一值
        df = df[df['contract_date'].str.len() == 6] 
        df = df.groupby('date').first().reset_index()
        df = df.rename(columns={"date": "Date", "close": "TX_Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df[['TX_Close']].apply(pd.to_numeric, errors='coerce')
    except:
        return pd.DataFrame()

# ==========================================
# 核心策略模組
# ==========================================
def run_basis_strategy(df_target, df_market, df_futures):
    df = df_target.copy()
    
    # 嚴格對齊日期，避免不同步導致的運算錯誤
    df_market = df_market.reindex(df.index).ffill()
    df_futures = df_futures.reindex(df.index).ffill()
    
    # --- 1. 環境濾網：正價差偵測 ---
    df['Basis'] = df_futures['TX_Close'] - df_market['Close']
    df['Contango_Warning'] = df['Basis'] > 0 
    
    # --- 2. 微觀觸發：價格破底 ---
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Contango'] = df['Contango_Warning'].shift(1)
    df['Break_Low_Exit'] = df['Prev_Contango'] & (df['Close'] < df['Prev_Low'])
    
    # --- 3. 進場與總經防禦邏輯 ---
    df['CCI40'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=40)
    df['C1_Entry'] = df['CCI40'] < -150
    
    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'], step=0.015, max_step=0.3)
    df['SAR'] = psar.psar()
    df['C2_Entry'] = (df['Close'] > df['SAR']) & (df['Close'].shift(1) <= df['SAR'].shift(1))
    
    df['Market_MA20'] = df_market['Close'].rolling(20).mean()
    df['Market_Bear'] = df_market['Close'] < df['Market_MA20']
    
    # --- 4. 優先權狀態機 (每日收盤後判定) ---
    df['Position'] = 0
    current_position = 0
    
    for i in range(1, len(df)):
        c1 = df['C1_Entry'].iloc[i]
        c2 = df['C2_Entry'].iloc[i]
        bear = df['Market_Bear'].iloc[i]
        take_profit = df['Break_Low_Exit'].iloc[i]
        
        # 優先權 1: 強制清倉 (破底停利 或 大盤空頭)
        if take_profit or bear:
            current_position = 0
        # 優先權 2: 放行買進 (有進場訊號 且 大盤多頭)
        elif (c1 or c2) and not bear:
            current_position = 1
            
        df.iat[i, df.columns.get_loc('Position')] = current_position

    # 產生交易動作標籤
    df['Position_Shift'] = df['Position'].diff()
    df['Action'] = ""
    df.loc[df['Position_Shift'] == 1, 'Action'] = "BUY (Next Open)"
    df.loc[df['Position_Shift'] == -1, 'Action'] = "SELL (Next Open)"
    
    return df

# ==========================================
# 執行與圖表渲染模組
# ==========================================
if st.sidebar.button("執行正價差矩陣運算"):
    if not finmind_token:
        st.error("請在左側輸入 FinMind API Token。")
    else:
        with st.spinner('正在獲取 ETF、大盤現貨與台指期貨資料並運算模型...'):
            df_target = fetch_stock_data("00631L", lookback_years, finmind_token)
            df_market = fetch_stock_data("TAIEX", lookback_years, finmind_token)
            df_futures = fetch_futures_data(lookback_years, finmind_token)
            
            if not df_target.empty and not df_market.empty and not df_futures.empty:
                result_df = run_basis_strategy(df_target, df_market, df_futures)
                
                # 控制圖表顯示天數
                plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='00631L'))
                
                # 標記正價差警報
                contango_days = plot_df[plot_df['Contango_Warning']]
                fig.add_trace(go.Scatter(x=contango_days.index, y=contango_days['High']*1.01, mode='markers',
                                         marker=dict(symbol='square', size=4, color='orange'), name='正價差環境'))
                
                # 標記買賣點
                buys = plot_df[plot_df['Position_Shift'] == 1]
                sells = plot_df[plot_df['Position_Shift'] == -1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers',
                                         marker=dict(symbol='triangle-up', size=12, color='green'), name='觸發買進'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers',
                                         marker=dict(symbol='triangle-down', size=12, color='red'), name='強制平倉'))
                
                fig.update_layout(
                    title=f"00631L: 正價差環境與破底觸發策略 (顯示 {plot_days if plot_days > 0 else '全部'} 交易日)", 
                    xaxis_title="日期", yaxis_title="價格", height=650,
                    xaxis=dict(type='date') # 確保 X 軸日期不失真
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("訊號監控 (近期狀態)")
                display_cols = ['Close', 'Basis', 'Contango_Warning', 'Market_Bear', 'Break_Low_Exit', 'Position', 'Action']
                
                # 使用 .map() 取代已棄用的 .applymap()
                st.dataframe(result_df[display_cols].tail(15).style.map(
                    lambda x: 'background-color: #ffcccc' if x == 0 else ('background-color: #ccffcc' if x == 1 else ''),
                    subset=['Position']
                ))
            else:
                st.error("獲取資料失敗，請確認 API Token 是否正確或稍後再試。")