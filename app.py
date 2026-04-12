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
st.title("00631L.TW 策略分析")

st.sidebar.subheader("資料源設定")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")
lookback_years = st.sidebar.number_input("回測年數", min_value=1, max_value=10, value=5)
plot_days = st.sidebar.slider("圖表顯示天數 (0為顯示全部)", 0, 1500, 0, step=50)

ticker = "00631L"

st.sidebar.markdown("---")
st.sidebar.subheader("資金回測與摩擦成本設定")
bt_start_date = st.sidebar.date_input("回測起始日", datetime.strptime('2024-01-01', '%Y-%m-%d'))
initial_capital = st.sidebar.number_input("起始投入資金 (NTD)", min_value=10000, value=100000, step=10000)
# 預設台股手續費 0.1425%，ETF 交易稅 0.1%
fee_rate = st.sidebar.number_input("券商單邊手續費率 (%)", value=0.1425, format="%.4f") / 100
tax_rate = st.sidebar.number_input("ETF 賣出交易稅率 (%)", value=0.1, format="%.3f") / 100

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

    # --- 獨立觀測指標：實質期現貨價差 (融合三維度狀態與季節性遮蔽) ---
    # 確保期貨索引對齊
    df_futures = df_futures.reindex(df.index).ffill()
    df['Basis'] = df_futures['Futures_Close'] - df_taiex['Close']
    
    # 🚨 關鍵修正：引入 5 日平滑均線，消除單日籌碼雜訊 🚨
    df['Smooth_Basis'] = df['Basis'].rolling(window=5).mean()
    
    df['Month'] = df.index.month
    df['Is_Dividend_Season'] = df['Month'].isin([6, 7, 8])
    
    def categorize_basis(row):
        b = row['Smooth_Basis']  # 改用「平滑後的價差」來進行狀態判定
        is_div = row['Is_Dividend_Season']
        
        if pd.isna(b): return "數據不足"
        
        if is_div:
            # 旺季 (6-8月)
            if b >= 20: return "極端正價差"
            elif b >= 0: return "微幅正價差"
            else: return "假性逆價差"
        else:
            # 非旺季 (9月~隔年5月)
            if b >= 40: return "極端正價差"
            elif b >= 5: return "微幅正價差"
            elif b < 0: return "實質逆價差"
            else: return "平水雜訊"
            
    df['Basis_State'] = df.apply(categorize_basis, axis=1)
    
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
# 資金回測模組 (導入真實稅費與整股交易模型)
# ==========================================
def calculate_equity_curve(df, start_date, initial_capital, fee_rate, tax_rate):
    mask = df.index >= pd.to_datetime(start_date)
    if not mask.any(): return pd.DataFrame()
    
    btest_df = df.loc[mask].copy()
    
    # 判斷進入回測期第一天的初始狀態 (繼承 T-1 的決策)
    idx_start = df.index.get_indexer([pd.to_datetime(start_date)], method='bfill')[0]
    initial_pos = df['Position'].iloc[idx_start - 1] if idx_start > 0 else 0
    
    cash = initial_capital
    shares = 0
    
    # 若初始狀態為滿倉，第一天開盤即建倉
    if initial_pos == 1:
        first_open = btest_df['Adj_Open'].iloc[0]
        # 嚴格整股計算：買進需支付手續費
        shares = np.floor(cash / (first_open * (1 + fee_rate)))
        if shares > 0:
            cost = shares * first_open
            fee = cost * fee_rate
            cash = cash - cost - fee
            
    equity = []
    
    for i in range(len(btest_df)):
        today_open = btest_df['Adj_Open'].iloc[i]
        today_close = btest_df['Adj_Close'].iloc[i]
        
        # 檢查【昨天】的訊號，決定【今天】開盤的動作
        if i > 0:
            prev_action = btest_df['Action'].iloc[i-1]
            if prev_action == "BUY" and cash > 0:
                # 買進：扣除手續費 (使用 np.floor 確保只買整數股)
                max_shares = np.floor(cash / (today_open * (1 + fee_rate)))
                if max_shares > 0:
                    cost = max_shares * today_open
                    fee = cost * fee_rate
                    cash = cash - cost - fee
                    shares = max_shares
            elif prev_action == "SELL" and shares > 0:
                # 賣出：扣除手續費與 ETF 交易稅
                gross_proceeds = shares * today_open
                fee = gross_proceeds * fee_rate
                tax = gross_proceeds * tax_rate
                cash = cash + gross_proceeds - fee - tax
                shares = 0
                
        # 結算今天盤後的總淨值 (現金 + 股票還原市值)
        current_value = cash + (shares * today_close)
        equity.append(current_value)
        
    btest_df['Equity'] = equity
    btest_df['Drawdown'] = (btest_df['Equity'] / btest_df['Equity'].cummax()) - 1
    
    return btest_df

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
                result_df = run_basis_strategy(df_target, df_taiex, df_otc, df_futures)
                
                # ==========================================
                # [第一區] 戰略全貌：價格行為與訊號觸發圖
                # ==========================================
                st.header("區域一：歷史軌跡與訊號圖表")
                plot_df = result_df.tail(plot_days) if plot_days > 0 else result_df
                
                fig = go.Figure()
                
                # --- 視覺升級 1：繪製持倉區間底色 (解決視覺斷層) ---
                in_position = False
                start_date = None

                for date, row in plot_df.iterrows():
                    if row['Position'] == 1 and not in_position:
                        start_date = date
                        in_position = True
                    elif row['Position'] == 0 and in_position:
                        # 畫出半透明淺綠色區塊，將整筆交易連貫起來
                        fig.add_vrect(x0=start_date, x1=date, fillcolor="rgba(0, 200, 0, 0.15)", layer="below", line_width=0)
                        in_position = False

                # 處理圖表最右側最後一天仍持倉的邊界狀況
                if in_position:
                    fig.add_vrect(x0=start_date, x1=plot_df.index[-1], fillcolor="rgba(0, 200, 0, 0.15)", layer="below", line_width=0)

                # --- 視覺升級 2：繪製 K 線圖 ---
                fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='K線(實際報價)'))
                
                # --- 視覺升級 3：強化買賣點標記 ---
                buys = plot_df[plot_df['Position_Shift'] == 1]
                sells = plot_df[plot_df['Position_Shift'] == -1]
                
                # 加大 size 並加上 line 邊框，讓訊號更立體不被 K 線吃掉
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
                                         marker=dict(symbol='triangle-up', size=16, color='lime', line=dict(color='darkgreen', width=2)), name='買進 (建倉)'))
                fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
                                         marker=dict(symbol='triangle-down', size=16, color='red', line=dict(color='darkred', width=2)), name='賣出 (清倉)'))
                
                # 關閉 xaxis_rangeslider_visible 釋放圖表高度空間
                fig.update_layout(title=f"{ticker} 雙軌防禦矩陣 (含持倉區間視覺化)", xaxis_title="日期", yaxis_title="價格", height=600,
                                  xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider() # 強制分隔線
                
                # ==========================================
                # [第二區] 執行指令：次日操作監控表
                # ==========================================
                st.header("區域二：次日開盤執行監控表")
                view_df = result_df.tail(15).copy()
                view_df.index = view_df.index.strftime('%Y-%m-%d')
                view_df.index.name = '日期'

                # 狀態文字轉換
                view_df['大盤與廣度(C4)'] = view_df.apply(lambda row: "空頭/流血" if row['C4_Exit'] else "健康", axis=1)
                view_df['趨勢平滑濾網(C6)'] = view_df.apply(lambda row: "趨勢延續" if row['C6_Trend_Active'] else "盤整阻斷", axis=1)
                
                # 完美映射表格邏輯至 UI 儀表板
                def map_basis_ui(state):
                    if state == "極端正價差": return "⚠️ 情緒過熱 (不宜追高)"
                    elif state == "微幅正價差": return "🔥 軋空起手 (吃主升段)"
                    elif state == "實質逆價差": return "🛡️ 轉倉紅利 (長線優質)"
                    elif state == "假性逆價差": return "❄️ 除息干擾 (忽略價差)"
                    else: return "⚖️ 價差平水 (動能不明)"
                    
                view_df['籌碼觀測'] = view_df['Basis_State'].apply(map_basis_ui)

                # 前端降噪：隱藏內部運算欄位，只顯示決策資訊與實際報價
                display_cols = ['Close', 'C1_Entry', 'C2_Entry', '大盤與廣度(C4)', 'C5_Exit', '趨勢平滑濾網(C6)', '籌碼觀測', 'Position', 'Action']
                view_df = view_df[display_cols]
                
                view_df = view_df.rename(columns={'Close': '實際報價(下單看這)'})

                st.dataframe(view_df.style.map(
                    lambda x: 'background-color: #ffcccc' if x == 0 else ('background-color: #ccffcc' if x == 1 else ''),
                    subset=['Position']
                ), use_container_width=True)
                
                st.divider() # 強制分隔線

                # ==========================================
                # [第三區] 戰略評估：實盤資金權益曲線
                # ==========================================
                st.header("區域三：資金績效模擬 (T+1開盤執行, 含稅費)")
                # 傳入側邊欄設定的動態參數
                btest_df = calculate_equity_curve(result_df, start_date=bt_start_date, 
                                                  initial_capital=initial_capital, 
                                                  fee_rate=fee_rate, tax_rate=tax_rate)
                
                if not btest_df.empty:
                    final_equity = btest_df['Equity'].iloc[-1]
                    max_dd = btest_df['Drawdown'].min() * 100
                    total_return = ((final_equity / initial_capital) - 1) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("最終帳戶淨值 (NTD)", f"${final_equity:,.0f}")
                    col2.metric("區間總報酬率", f"{total_return:.2f}%")
                    col3.metric("最大歷史回檔 (MDD)", f"{max_dd:.2f}%")
                    
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=btest_df.index, y=btest_df['Equity'], 
                                                mode='lines', name='帳戶總淨值', line=dict(color='gold', width=2)))
                    
                    b_points = btest_df[btest_df['Action'].shift(1) == 'BUY']
                    s_points = btest_df[btest_df['Action'].shift(1) == 'SELL']
                    
                    fig_eq.add_trace(go.Scatter(x=b_points.index, y=b_points['Equity'], mode='markers',
                                             marker=dict(symbol='triangle-up', size=10, color='green'), name='開盤買進(扣手續費)'))
                    fig_eq.add_trace(go.Scatter(x=s_points.index, y=s_points['Equity'], mode='markers',
                                             marker=dict(symbol='triangle-down', size=10, color='red'), name='開盤賣出(扣稅費)'))
                    
                    fig_eq.update_layout(title=f"系統權益曲線 (起始本金: {initial_capital:,.0f})", xaxis_title="日期", yaxis_title="淨值 (NTD)", height=400)
                    st.plotly_chart(fig_eq, use_container_width=True)
            
            # --- 這是你漏掉的除錯警報器，必須嚴格對齊 if not error_msgs: ---
            else:
                st.error("🚨 核心資料鏈斷裂，策略中止執行。")
                for msg in error_msgs:
                    st.error(f"-> {msg}")
                st.warning("若持續發生此錯誤，極可能是 FinMind API 每小時請求次數已達上限，請稍候再試。")