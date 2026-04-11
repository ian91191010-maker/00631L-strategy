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
st.title("00631L.TW 進出場策略")

st.sidebar.subheader("資料源設定")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")
lookback_years = st.sidebar.number_input("回測年數", min_value=1, max_value=10, value=5)
plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)

ticker = "00631L"

# ==========================================
# 資料獲取模組 
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol, years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date, "end_date": end_date, "token": token}
    try:
        res = requests.get(url, params=parameter, timeout=15)
        
        # 捕捉 HTTP 狀態碼錯誤 (如 429 Too Many Requests)
        if res.status_code != 200:
            st.warning(f"[{symbol}] API HTTP 錯誤: 狀態碼 {res.status_code}")
            return pd.DataFrame()
            
        json_data = res.json()
        # 捕捉 FinMind 自定義的錯誤訊息
        if json_data.get("msg") != "success":
            st.warning(f"[{symbol}] API 拒絕請求: {json_data.get('msg')}")
            return pd.DataFrame()
            
        df = pd.DataFrame(json_data.get("data", []))
        if df.empty: 
            st.warning(f"[{symbol}] 請求成功，但回傳資料為空值，請確認代碼是否正確。")
            return df
            
        # ... (此處保留原有的改名、轉型與分割乘數還原運算) ...
        df = df.rename(columns={"date": "Date", "open": "Open", "max": "High", "min": "Low", "close": "Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        df['Adj_Open'] = df['Open']
        df['Adj_High'] = df['High']
        df['Adj_Low'] = df['Low']
        df['Adj_Close'] = df['Close']
        
        # 針對 00631L 的股票分割事件進行官方比例寫死 (Hardcode)
        # 注意：請將 '2026-03-31' 替換為正確的歷史年份，並確認 split_ratio
        split_date = pd.to_datetime('2026-03-31') 
        split_ratio = 22.0
        
        mask = df.index < split_date
        
        df.loc[mask, 'Adj_Open'] = df.loc[mask, 'Open'] / split_ratio
        df.loc[mask, 'Adj_High'] = df.loc[mask, 'High'] / split_ratio
        df.loc[mask, 'Adj_Low'] = df.loc[mask, 'Low'] / split_ratio
        df.loc[mask, 'Adj_Close'] = df.loc[mask, 'Close'] / split_ratio
            
        return df
    except Exception as e:
        st.warning(f"[{symbol}] 發生系統異常: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_futures_data(years, token):
    if not token: return pd.DataFrame()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanFuturesDaily", 
        "data_id": "TX", # 台股期貨
        "start_date": start_date, 
        "end_date": end_date, 
        "token": token
    }
    
    try:
        res = requests.get(url, params=parameter, timeout=15)
        if res.status_code != 200: return pd.DataFrame()
        
        df = pd.DataFrame(res.json().get("data", []))
        if df.empty: return df
        
        # 期貨資料清洗：只取每天的「近月合約」(通常是該日期下 contract_date 最近的)
        # 實務上最簡單的過濾法：以每日交易量 (trading_volume) 最大的合約為主力近月合約
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.sort_values(['date', 'volume'], ascending=[True, False])
        df = df.drop_duplicates(subset=['date'], keep='first') # 每天只保留交易量最大的一筆
        
        df = df.rename(columns={"date": "Date", "close": "Futures_Close"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Futures_Close'] = pd.to_numeric(df['Futures_Close'], errors='coerce')
        
        return df[['Futures_Close']]
    except Exception as e:
        st.warning(f"[TX期貨] 發生系統異常: {str(e)}")
        return pd.DataFrame()

# ==========================================
# 核心策略模組 (狀態機與平滑邏輯修復)
# ==========================================
def run_basis_strategy(df_target, df_taiex, df_otc, df_futures):
    df = df_target.copy()
    # 確保索引對齊，避免跨市場資料錯位
    df_taiex = df_taiex.reindex(df.index).ffill()
    df_otc = df_otc.reindex(df.index).ffill()
    
    # --- 1. 戰術進場觸發 (C1 & C2) ---
    df['CCI40'] = ta.trend.cci(df['Adj_High'], df['Adj_Low'], df['Adj_Close'], window=40)
    df['C1_Entry'] = df['CCI40'] < -150
    
    psar = ta.trend.PSARIndicator(df['Adj_High'], df['Adj_Low'], df['Adj_Close'], step=0.015, max_step=0.3)
    df['SAR'] = psar.psar()
    df['C2_Entry'] = (df['Adj_Close'] > df['SAR']) & (df['Adj_Close'].shift(1) <= df['SAR'].shift(1))

    # 確保期貨索引對齊
    df_futures = df_futures.reindex(df.index).ffill()
    
    df['Basis'] = df_futures['Futures_Close'] - df_taiex['Close']
    df['Positive_Basis'] = df['Basis'] > 0  # 純粹作為觀測標記
    
    # --- 2. 出場訊號 (C4 & C5) ---
    # C4 熊市偵測: 大盤週月線死叉 OR 櫃買(廣度替代)週月線死叉
    taiex_ma5 = df_taiex['Close'].rolling(5).mean()
    taiex_ma20 = df_taiex['Close'].rolling(20).mean()
    otc_ma5 = df_otc['Close'].rolling(5).mean()
    otc_ma20 = df_otc['Close'].rolling(20).mean()
    
    df['C4_Exit'] = (taiex_ma5 < taiex_ma20) | (otc_ma5 < otc_ma20)

    # C5 急跌保護
    df['Daily_Ret'] = df['Adj_Close'].pct_change()
    df['Rolling_5D_Ret'] = df['Adj_Close'].pct_change(periods=5).shift(1)
    cond_extreme_drop = df['Daily_Ret'] < -0.035
    cond_compound_drop = (df['Daily_Ret'] < -0.025) & (df['Rolling_5D_Ret'] < -0.02)
    df['C5_Exit'] = cond_extreme_drop | cond_compound_drop
    
    # --- 3. 趨勢濾網 (C6 VHF 平滑機制修復) ---
    window_vhf = 35
    highest_close = df['Adj_Close'].rolling(window_vhf).max()
    lowest_close = df['Adj_Close'].rolling(window_vhf).min()
    abs_diff = df['Adj_Close'].diff().abs().rolling(window_vhf).sum()
    df['VHF'] = (highest_close - lowest_close) / abs_diff
    
    # 遲滯現象 (Hysteresis) 實作：開3日/關3日
    df['VHF_Over_02'] = df['VHF'] > 0.2
    df['C6_Trend_Active'] = False
    
    trend_state = False
    c6_states = []
    for val in df['VHF_Over_02'].rolling(3):
        if len(val) == 3:
            if val.sum() == 3:
                trend_state = True
            elif val.sum() == 0:
                trend_state = False
        c6_states.append(trend_state)
    df['C6_Trend_Active'] = c6_states

    # --- 4. 倉位狀態機 (嚴格實作圖表優先權) ---
    # ①預設(1) -> ②出場 -> ③進場(覆蓋出場) -> ④濾網(最終否決)
    df['Position'] = 0
    
    for i in range(1, len(df)):
        c1 = df['C1_Entry'].iloc[i]
        c2 = df['C2_Entry'].iloc[i]
        c4 = df['C4_Exit'].iloc[i]
        c5 = df['C5_Exit'].iloc[i]
        c6 = df['C6_Trend_Active'].iloc[i]
        
        # 步驟 1: 預設滿倉
        pos = 1 
        # 步驟 2: 防禦機制剔除 (標準邏輯：僅 C4, C5)
        if c4 or c5: pos = 0
        # 步驟 3: 極端訊號抄底 (標準邏輯：僅 C1, C2)
        if c1 or c2: pos = 1
        # 步驟 4: 趨勢濾網最終裁決
        if not c6: pos = 0
            
        df.iat[i, df.columns.get_loc('Position')] = pos

    df['Position_Shift'] = df['Position'].diff()
    df['Action'] = ""
    df.loc[df['Position_Shift'] == 1, 'Action'] = "BUY"
    df.loc[df['Position_Shift'] == -1, 'Action'] = "SELL"
    
    return df

# ==========================================
# 執行與圖表渲染
# ==========================================
if st.sidebar.button("執行矩陣策略運算"):
    if not finmind_token:
        st.error("執行中止：尚未輸入 FinMind API Token。")
    else:
        with st.spinner('正在獲取多維度市場資料 (00631L, TAIEX, TPEx, TX)...'):
            df_target = fetch_stock_data(ticker, lookback_years, finmind_token)
            df_taiex = fetch_stock_data("TAIEX", lookback_years, finmind_token)
            df_otc = fetch_stock_data("TPEx", lookback_years, finmind_token) 
            df_futures = fetch_futures_data(lookback_years, finmind_token) # 新增期貨抓取
            
            error_msgs = []
            if df_target.empty: error_msgs.append(f"缺失 {ticker} 現貨資料")
            if df_taiex.empty: error_msgs.append("缺失 TAIEX (加權指數) 資料")
            if df_otc.empty: error_msgs.append("缺失 TPEx (櫃買指數) 資料")
            if df_futures.empty: error_msgs.append("缺失 TX (台股期貨) 資料")
            
            if not error_msgs:
                # 將 df_futures 傳入
                result_df = run_basis_strategy(df_target, df_taiex, df_otc, df_futures)
                # ... (下方繪圖與表格邏輯不變，但你可以把 C3_Entry 加進顯示欄位中) ...
                
                # --- 以下補回繪圖與表格渲染邏輯 ---
                plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='K線(原價)'))
                
                buys = plot_df[plot_df['Position_Shift'] == 1]
                sells = plot_df[plot_df['Position_Shift'] == -1]
                
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers',
                                         marker=dict(symbol='triangle-up', size=12, color='green'), name='策略買進'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers',
                                         marker=dict(symbol='triangle-down', size=12, color='red'), name='強制清倉'))
                
                fig.update_layout(title=f"{ticker} 雙軌防禦矩陣 (邏輯重構版)", xaxis_title="日期", yaxis_title="價格", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("核心矩陣狀態監控")
                view_df = result_df.tail(15).copy()
                view_df.index = view_df.index.strftime('%Y-%m-%d')
                view_df.index.name = '日期'

                # --- 雙軌顯示設定 ---
                # 將兩種價格都四捨五入，方便閱讀
                view_df['Close'] = view_df['Close'].round(2)
                view_df['Adj_Close'] = view_df['Adj_Close'].round(2)
                view_df['期現價差'] = view_df['Basis'].round(2)

                # 狀態文字轉換
                view_df['大盤與廣度(C4)'] = view_df.apply(lambda row: "空頭/流血" if row['C4_Exit'] else "健康", axis=1)
                view_df['趨勢平滑濾網(C6)'] = view_df.apply(lambda row: "趨勢延續" if row['C6_Trend_Active'] else "盤整阻斷", axis=1)
                view_df['籌碼觀測'] = view_df.apply(lambda row: "🔥 正價差" if row['Positive_Basis'] else "逆價差", axis=1)

                # 修正：將真實收盤價(Close)放回第一列，並保留還原價(Adj_Close)供對照
                display_cols = ['Close', 'C1_Entry', 'C2_Entry', '大盤與廣度(C4)', 'C5_Exit', '趨勢平滑濾網(C6)', '籌碼觀測', 'Position', 'Action']
                view_df = view_df[display_cols]
                
                # 終極防呆：強制重命名欄位，讓看盤直覺化
                view_df = view_df.rename(columns={
                    'Close': '實際報價(下單看這)',
                })

                st.dataframe(view_df.style.map(
                    lambda x: 'background-color: #ffcccc' if x == 0 else ('background-color: #ccffcc' if x == 1 else ''),
                    subset=['Position']
                ))