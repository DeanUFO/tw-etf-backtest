import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go

# ==========================================
# 網頁 UI 與版面設定
# ==========================================
st.set_page_config(page_title="台股 ETF 多重擂台賽 & AI 預測", layout="wide")
st.title("🏆 台股 ETF 策略回測 & AI 趨勢預測")
st.markdown("同時輸入多檔標的，找出最佳「定期定額 + 季線加碼」策略，並與終極大盤對決。")

# --- 側邊欄：參數輸入區 ---
st.sidebar.header("⚙️ 參數設定")
st.sidebar.markdown("請用**逗號**分隔輸入 1~5 檔標的")
tickers_input = st.sidebar.text_input("目標策略代號 (最多5檔)", value="00878.TW, 00919.TW, 00929.TW")

st.sidebar.markdown("---")
st.sidebar.subheader("大盤基準設定")
benchmark_type = st.sidebar.radio(
    "選擇用來對照的大盤基準：",
    options=["加權指數 (^TWII) - 不含息", "報酬指數 (^TWII-TR) - 含息總報酬", "台灣50 (0050.TW) - 實體ETF"],
    index=1  # 預設為最嚴格的「含息報酬指數」
)

# 解析大盤代碼
if "TWII-TR" in benchmark_type:
    benchmark_ticker = "^TWII-TR"
elif "TWII" in benchmark_type:
    benchmark_ticker = "^TWII"
else:
    benchmark_ticker = "0050.TW"

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-05-01"))

st.sidebar.markdown("---")
initial_cash = st.sidebar.number_input("初始備用現金池 (元)", min_value=0, value=100000, step=10000)
dca_amount = st.sidebar.number_input("每月定期定額金額 (元)", min_value=1000, value=10000, step=1000)
bonus_amount = st.sidebar.number_input("跌破季線加碼金額 (元)", min_value=0, value=20000, step=1000)
fee_discount = st.sidebar.slider("券商手續費折扣", min_value=0.1, max_value=1.0, value=0.6, step=0.01)

# --- 預測趨勢設定 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 未來趨勢預測")
forecast_days = st.sidebar.slider("預測未來天數", min_value=30, max_value=365, value=180, step=30)

def calculate_fee(cost):
    fee = cost * 0.001425 * fee_discount
    return max(20, np.floor(fee))

# ==========================================
# 核心回測與預測邏輯
# ==========================================
if st.sidebar.button("🚀 開始擂台賽與預測", type="primary"):
    
    raw_tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
    if len(raw_tickers) > 5:
        st.sidebar.error("⚠️ 最多請輸入 5 檔標的。")
        st.stop()
    if not raw_tickers:
        st.sidebar.error("⚠️ 請至少輸入 1 檔標的。")
        st.stop()

    with st.spinner(f"正在下載數據並與 {benchmark_ticker} 執行多重對決中..."):
        
        market_data = {}
        all_dates = None 

        try:
            # 1. 抓取大盤對照組數據
            # 若為指數 (^TWII, ^TWII-TR)，通常沒有配息資料，這部分會自動避開
            df_bench_raw = yf.download(benchmark_ticker, start=start_date, end=end_date)
            df_bench_raw.index = pd.to_datetime(df_bench_raw.index).tz_localize(None)
            all_dates = df_bench_raw.index
            
            bench_close = df_bench_raw['Close'].squeeze()
            div_bench = yf.Ticker(benchmark_ticker).dividends
            if not div_bench.empty:
                div_bench.index = pd.to_datetime(div_bench.index).tz_localize(None)
                div_bench = div_bench.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
            
            market_data[benchmark_ticker] = {
                'Close': bench_close,
                'Div': div_bench
            }

            # 2. 抓取所有目標標的數據
            for ticker in raw_tickers:
                df_raw = yf.download(ticker, start=start_date, end=end_date)
                if df_raw.empty:
                    st.error(f"⚠️ 找不到 {ticker} 的數據。")
                    st.stop()
                    
                df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
                
                div_raw = yf.Ticker(ticker).dividends
                if not div_raw.empty:
                    div_raw.index = pd.to_datetime(div_raw.index).tz_localize(None)
                    div_raw = div_raw.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
                
                close_series = df_raw['Close'].squeeze()
                ma60_series = close_series.rolling(window=60).mean()
                
                market_data[ticker] = {
                    'Close': close_series,
                    'MA60': ma60_series,
                    'Div': div_raw
                }
                
        except Exception as e:
            st.error(f"下載數據時發生錯誤: {e}")
            st.stop()

        # --- 初始化所有帳戶狀態 ---
        accounts = {}
        accounts[benchmark_ticker] = {'cash_pool': 0, 'shares': 0, 'accumulated_principal': 0}
        is_first_day = True

        for ticker in raw_tickers:
            accounts[ticker] = {
                'cash_pool': initial_cash, 'shares': 0, 
                'accumulated_principal': 0, 'bonus_months': set()
            }

        chart_data = []
        aligned_df = pd.DataFrame(index=all_dates)
        aligned_df['YearMonth'] = aligned_df.index.to_period('M')
        first_trading_days = aligned_df.groupby('YearMonth').head(1).index

        # --- 3. 多重雙軌回測迴圈 ---
        for date, row in aligned_df.iterrows():
            current_month = row['YearMonth']
            
            if date not in market_data[benchmark_ticker]['Close'].index:
                continue
            p_bench = float(market_data[benchmark_ticker]['Close'].loc[date])
            if pd.isna(p_bench): continue

            if is_first_day:
                # 指數交易通常不計手續費，實體 ETF (0050) 才計
                fee = calculate_fee(initial_cash) if benchmark_ticker == "0050.TW" else 0
                accounts[benchmark_ticker]['shares'] += (initial_cash - fee) / p_bench
                is_first_day = False

            div_b = market_data[benchmark_ticker]['Div']
            if not div_b.empty and date in div_b.index:
                accounts[benchmark_ticker]['cash_pool'] += accounts[benchmark_ticker]['shares'] * float(div_b.loc[date])

            daily_chart_record = {'Date': date, '累積投入本金 (基準)': 0}
            
            for ticker in raw_tickers:
                if date not in market_data[ticker]['Close'].index:
                    continue
                    
                p_main = float(market_data[ticker]['Close'].loc[date])
                ma60 = float(market_data[ticker]['MA60'].loc[date]) if pd.notna(market_data[ticker]['MA60'].loc[date]) else None
                if pd.isna(p_main): continue
                
                acct = accounts[ticker]

                div_m = market_data[ticker]['Div']
                if not div_m.empty and date in div_m.index:
                    acct['cash_pool'] += acct['shares'] * float(div_m.loc[date])

                if date in first_trading_days:
                    acct['accumulated_principal'] += dca_amount
                    acct['cash_pool'] += dca_amount
                    acct['shares'] += (dca_amount - calculate_fee(dca_amount)) / p_main
                    acct['cash_pool'] -= dca_amount
                    daily_chart_record['累積投入本金 (基準)'] = acct['accumulated_principal'] + initial_cash

                if ma60 and p_main < ma60 and current_month not in acct['bonus_months']:
                    if acct['cash_pool'] >= bonus_amount:
                        acct['shares'] += (bonus_amount - calculate_fee(bonus_amount)) / p_main
                        acct['cash_pool'] -= bonus_amount
                        acct['bonus_months'].add(current_month)

                daily_chart_record[f"{ticker}"] = (acct['shares'] * p_main) + acct['cash_pool']

            if date in first_trading_days:
                b_acct = accounts[benchmark_ticker]
                b_acct['accumulated_principal'] += dca_amount
                b_acct['cash_pool'] += dca_amount
                fee = calculate_fee(dca_amount) if benchmark_ticker == "0050.TW" else 0
                b_acct['shares'] += (dca_amount - fee) / p_bench
                b_acct['cash_pool'] -= dca_amount
                
            daily_chart_record[f"對照組 ({benchmark_ticker})"] = (accounts[benchmark_ticker]['shares'] * p_bench) + accounts[benchmark_ticker]['cash_pool']

            if daily_chart_record['累積投入本金 (基準)'] > 0:
                chart_data.append(daily_chart_record)

        # --- 4. 結算與網頁呈現 ---
        total_input = chart_data[-1]['累積投入本金 (基準)']
        
        st.subheader("📊 擂台賽績效排行")
        st.markdown(f"**總投入資金基準：** `${total_input:,.0f}`")
        
        results = []
        final_bench_asset = chart_data[-1][f"對照組 ({benchmark_ticker})"]
        bench_roi = (final_bench_asset - total_input) / total_input * 100
        
        for ticker in raw_tickers:
            if ticker in chart_data[-1]:
                final_asset = chart_data[-1][ticker]
                roi = (final_asset - total_input) / total_input * 100
                diff_to_bench = final_asset - final_bench_asset
                results.append({
                    "標的": ticker,
                    "最終總資產": final_asset,
                    "報酬率 (%)": roi,
                    "勝過大盤金額": diff_to_bench
                })
        
        df_results = pd.DataFrame(results).sort_values(by="最終總資產", ascending=False).reset_index(drop=True)
        
        st.info(f"📈 **【基準對照組】 {benchmark_type}** | 最終總資產: ${final_bench_asset:,.0f} | 報酬率: {bench_roi:.2f}%")
        
        cols = st.columns(len(df_results))
        for idx, row in df_results.iterrows():
            with cols[idx]:
                diff_text = f"贏大盤 ${row['勝過大盤金額']:,.0f}" if row['勝過大盤金額'] > 0 else f"輸大盤 ${abs(row['勝過大盤金額']):,.0f}"
                st.metric(
                    label=f"🏆 第 {idx+1} 名: {row['標的']}", 
                    value=f"${row['最終總資產']:,.0f}", 
                    delta=f"{row['報酬率 (%)']:.2f}% | {diff_text}",
                    delta_color="normal" if row['勝過大盤金額'] > 0 else "inverse"
                )

        st.markdown("---")
        st.subheader("📈 總資產成長曲線對決")
        df_chart = pd.DataFrame(chart_data).set_index('Date')
        df_chart = df_chart.fillna(method='ffill').fillna(0)
        
        cols_to_plot = [c for c in df_chart.columns if c != '累積投入本金 (基準)']
        st.line_chart(df_chart[cols_to_plot])

        # ==========================================
        # 🔮 未來趨勢預測模組 (Prophet)
        # ==========================================
        st.markdown("---")
        st.subheader(f"🤖 AI 趨勢預測 (未來 {forecast_days} 天)")
        st.markdown("使用 Meta Prophet 模型，基於歷史波動預測未來股價走勢區間。(註：僅供參考，非保證獲利)")

        tabs = st.tabs(raw_tickers)

        for i, ticker in enumerate(raw_tickers):
            with tabs[i]:
                df_history = market_data[ticker]['Close'].dropna().reset_index()
                df_history.columns = ['ds', 'y'] 

                if len(df_history) < 100:
                    st.warning(f"⚠️ {ticker} 歷史數據少於 100 天，預測準確度過低，建議略過。")
                    continue
                
                with st.spinner(f"正在運算 {ticker} 的未來趨勢..."):
                    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
                    m.fit(df_history)

                    future = m.make_future_dataframe(periods=forecast_days)
                    forecast = m.predict(future)

                    trace_actual = go.Scatter(
                        x=df_history['ds'], y=df_history['y'],
                        mode='lines', name='歷史實際股價',
                        line=dict(color='black', width=1.5)
                    )
                    
                    trace_pred = go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', name='AI 預測中線',
                        line=dict(color='blue', width=2, dash='dot')
                    )
                    
                    trace_upper = go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'],
                        mode='lines', fill=None, line=dict(color='rgba(0,0,255,0)') , showlegend=False
                    )
                    trace_lower = go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'],
                        mode='lines', fill='tonexty', fillcolor='rgba(0,0,255,0.2)', 
                        line=dict(color='rgba(0,0,255,0)'), name='80% 信賴區間'
                    )

                    layout = go.Layout(
                        title=f"{ticker} 股價趨勢預測",
                        xaxis=dict(title="日期"),
                        yaxis=dict(title="股價 (元)"),
                        hovermode='x unified'
                    )

                    fig = go.Figure(data=[trace_actual, trace_upper, trace_lower, trace_pred], layout=layout)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    last_actual = df_history['y'].iloc[-1]
                    future_target = forecast['yhat'].iloc[-1]
                    upside = (future_target - last_actual) / last_actual * 100
                    
                    st.write(f"**預測摘要：**")
                    st.write(f"目前最後收盤價： **${last_actual:.2f}**")
                    st.write(f"{forecast_days} 天後預測中位數： **${future_target:.2f}** (預期變化： **{upside:.2f}%**)")