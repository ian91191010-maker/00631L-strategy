import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==========================================
# 網頁 UI 設定
# ==========================================
st.set_page_config(page_title="正價差與價格行為防禦策略", layout="wide")
st.title("00631L.TW 雙軌防禦矩陣")

st.sidebar.subheader("資料源設定")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")
lookback_years = st.sidebar.number_input("回測年數", min_value=1, max_value=10, value=5)
plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)

ticker = "00631L"

# ==========================================
# 資料獲取模組 (必須抓取三個維度)
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
        # 篩選近月合約
        df = df[df['contract_date'].str.len() == 6] 
        df = df.groupby('date').first().reset_index()
        df = df.rename(columns={"date": "Date", "close": "TX_Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df[['TX_Close']].apply(pd.to_numeric, errors='coerce')
    except:
        return pd.DataFrame()

# ==========================================
# 核心策略模組 (雙軌斜率：20MA 阻斷 + 60MA 逃命)
# ==========================================
def run_basis_strategy(df_target, df_market, df_futures):
    df = df_target.copy()
    
    # 嚴格對齊日期
    df_market = df_market.reindex(df.index).ffill()
    df_futures = df_futures.reindex(df.index).ffill()
    
    # --- 1. 環境濾網：正價差偵測 ---
    df['Basis'] = df_futures['TX_Close'] - df_market['Close']
    df['Contango_Warning'] = df['Basis'] > 0 
    
    # --- 2. 微觀觸發：價格破底 ---
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Contango'] = df['Contango_Warning'].shift(1)
    df['Break_Low_Exit'] = df['Prev_Contango'] & (df['Close'] < df['Prev_Low'])
    
    # --- 3. 進場邏輯 (原始動能訊號) ---
    df['CCI40'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=40)
    df['C1_Entry'] = df['CCI40'] < -150
    
    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'], step=0.015, max_step=0.3)
    df['SAR'] = psar.psar()
    df['C2_Entry'] = (df['Close'] > df['SAR']) & (df['Close'].shift(1) <= df['SAR'].shift(1))
    
    # --- 4. 總經防禦：雙軌斜率運算 ---
    df['Market_MA20'] = df_market['Close'].rolling(20).mean()
    df['Market_MA60'] = df_market['Close'].rolling(60).mean()
    
    df['MA20_Slope'] = df['Market_MA20'] - df['Market_MA20'].shift(3)
    df['MA60_Slope'] = df['Market_MA60'] - df['Market_MA60'].shift(3)
    
    # 月線下彎 = 短期回檔 (阻斷進場)
    df['Market_Correction_20'] = df['MA20_Slope'] < 0 
    # 季線下彎 = 系統性熊市 (強制清倉)
    df['Market_Bear_60'] = df['MA60_Slope'] < 0
    
    # --- 5. 優先權狀態機 (三維度裁決) ---
    df['Position'] = 0
    current_position = 0
    
    for i in range(1, len(df)):
        c1 = df['C1_Entry'].iloc[i]
        c2 = df['C2_Entry'].iloc[i]
        bear_60 = df['Market_Bear_60'].iloc[i]
        correction_20 = df['Market_Correction_20'].iloc[i]
        take_profit = df['Break_Low_Exit'].iloc[i]
        
        # 優先權 1: 強制清倉 (系統性熊市 或 正價差破底)
        if bear_60 or take_profit:
            current_position = 0
            
        # 優先權 2: 放行買進 (有進場訊號 且 20MA 月線向上 且 60MA 季線向上)
        elif (c1 or c2) and not correction_20 and not bear_60:
            current_position = 1
            
        # 若發生 20MA 下彎但 60MA 向上：不進新單，但允許持有舊單 (繼承昨日狀態)
            
        df.iat[i, df.columns.get_loc('Position')] = current_position

    # 產生交易動作標籤
    df['Position_Shift'] = df['Position'].diff()
    df['Action'] = ""
    df.loc[df['Position_Shift'] == 1, 'Action'] = "BUY (Next Open)"
    df.loc[df['Position_Shift'] == -1, 'Action'] = "SELL (Next Open)"
    
    return df

# ==========================================
# 執行與圖表渲染
# ==========================================
if st.sidebar.button("執行矩陣策略運算"):
    if not finmind_token:
        st.error("執行中止：尚未輸入 FinMind API Token。")
    else:
        with st.spinner('正在獲取標的、大盤與期貨資料並執行運算...'):
            # 正確呼叫三個資料源
            df_target = fetch_stock_data(ticker, lookback_years, finmind_token)
            df_market = fetch_stock_data("TAIEX", lookback_years, finmind_token)
            df_futures = fetch_futures_data(lookback_years, finmind_token)
            
            if not df_target.empty and not df_market.empty and not df_futures.empty:
                # 正確將三個變數傳入新引擎
                result_df = run_basis_strategy(df_target, df_market, df_futures)
                
                # 圖表顯示區間控制
                plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='K線'))
                
                # 標示正價差警報區間
                contango_days = plot_df[plot_df['Contango_Warning']]
                fig.add_trace(go.Scatter(x=contango_days.index, y=contango_days['High']*1.01, mode='markers',
                                         marker=dict(symbol='square', size=5, color='orange'), name='正價差警報'))
                
                buys = plot_df[plot_df['Position_Shift'] == 1]
                sells = plot_df[plot_df['Position_Shift'] == -1]
                
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers',
                                         marker=dict(symbol='triangle-up', size=12, color='green'), name='濾網放行買進'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers',
                                         marker=dict(symbol='triangle-down', size=12, color='red'), name='強制清倉'))
                
                fig.update_layout(title=f"{ticker} 雙軌斜率與基差防禦矩陣", xaxis_title="日期", yaxis_title="價格", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("核心矩陣狀態監控")
                display_cols = ['Close', 'C1_Entry', 'C2_Entry', 'Market_Correction_20', 'Market_Bear_60', 'Contango_Warning', 'Break_Low_Exit', 'Position', 'Action']

                # 擷取最後 15 筆資料並獨立複製，避免改動到原始運算資料
                view_df = result_df[display_cols].tail(15).copy()

                # 1. 日期格式化：將 DatetimeIndex 的時間尾巴消掉，只保留 YYYY-MM-DD
                view_df.index = view_df.index.strftime('%Y-%m-%d')
                view_df.index.name = '日期'

                # 2. 欄位名稱中文化
                col_mapping = {
                    'Close': '收盤價',
                    'C1_Entry': 'CCI超賣',
                    'C2_Entry': 'SAR翻多',
                    'Market_Correction_20': '月線示警(阻斷)',
                    'Market_Bear_60': '季線示警(清倉)',
                    'Contango_Warning': '正價差警報',
                    'Break_Low_Exit': '破底觸發',
                    'Position': '倉位狀態',
                    'Action': '交易動作'
                }
                view_df = view_df.rename(columns=col_mapping)

                # 3. 結合數值格式化 (小數點後兩位) 與 顏色標記
                st.dataframe(view_df.style.format(
                    formatter={'收盤價': '{:.2f}'}  # 強制收盤價只顯示兩位小數
                ).map(
                    lambda x: 'background-color: #ffcccc' if x == 0 else ('background-color: #ccffcc' if x == 1 else ''),
                    subset=['倉位狀態']
                ))
            else:
                st.error("API 請求失敗或資料不足，請檢查 Token 或稍後再試。")