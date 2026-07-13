"""
STREAMLIT DASHBOARD - JSE PAIRS TRADING
North West University - Quant Project
Connects to FastAPI backend for live spread monitoring and backtesting
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="JSE Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS FOR PROFESSIONAL LOOK --------------------
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .signal-box {
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            margin: 0.5rem 0;
        }
        .signal-LONG_SPREAD {
            background-color: #D1FAE5;
            color: #065F46;
            border: 2px solid #10B981;
        }
        .signal-SHORT_SPREAD {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 2px solid #EF4444;
        }
        .signal-NEUTRAL {
            background-color: #F3F4F6;
            color: #374151;
            border: 2px solid #9CA3AF;
        }
        .metric-card {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1E3A8A;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #64748B;
        }
        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #E2E8F0;
        }
        .footer a {
            color: #1E3A8A;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR CONFIGURATION --------------------
st.sidebar.title("⚙️ Configuration")

# API Configuration
api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://localhost:8000"
)
st.sidebar.markdown("---")

# Pair selection
ticker1 = st.sidebar.text_input("Ticker 1", value="NPN.JO")
ticker2 = st.sidebar.text_input("Ticker 2", value="PRX.JO")
lookback = st.sidebar.slider("Lookback Days", 50, 500, 252)
entry_zscore = st.sidebar.number_input("Entry Z-Score", value=2.0, step=0.1)
exit_zscore = st.sidebar.number_input("Exit Z-Score", value=0.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Pre-set Pairs:**")
col1, col2 = st.sidebar.columns(2)
if col1.button("🇿🇦 Naspers/Prosus"):
    ticker1 = "NPN.JO"
    ticker2 = "PRX.JO"
if col2.button("🛢️ Sasol/Oil"):
    ticker1 = "SOL.JO"
    ticker2 = "CL=F"

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

# -------------------- DATA FETCHING FUNCTIONS --------------------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_spread_data(t1, t2, lookback, entry, exit):
    """Call FastAPI endpoint to get spread data"""
    try:
        response = requests.post(
            f"{api_url}/spread",
            json={
                "ticker1": t1,
                "ticker2": t2,
                "lookback_days": lookback,
                "entry_zscore": entry,
                "exit_zscore": exit
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI. Make sure it's running (python main.py).")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def run_backtest(payload):
    """Call the FastAPI /backtest endpoint. Returns the parsed JSON body, or None on error."""
    try:
        response = requests.post(f"{api_url}/backtest", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        st.error(f"Backtest error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI. Make sure it's running (python main.py).")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Backtest timed out. Try a shorter lookback or a coarser re-fit schedule.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# -------------------- MAIN DASHBOARD --------------------
st.markdown('<div class="main-header">🇿🇦 JSE Pairs Trading Dashboard</div>', unsafe_allow_html=True)
st.markdown(f"**Pair:** `{ticker1}` vs `{ticker2}` | **Lookback:** {lookback} days")

tab_live, tab_backtest = st.tabs(["📡 Live Monitor", "🧪 Backtest"])

# ==================== LIVE MONITOR TAB ====================
with tab_live:
    spread_data = fetch_spread_data(ticker1, ticker2, lookback, entry_zscore, exit_zscore)

    if spread_data:
        # -------------------- TOP METRICS ROW --------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Spread</div>
                    <div class="metric-value">{spread_data.get('current_spread', 0):.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Z-Score</div>
                    <div class="metric-value">{spread_data.get('current_zscore', 0):.3f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Hedge Ratio</div>
                    <div class="metric-value">{spread_data.get('hedge_ratio', 0):.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Half-Life (Days)</div>
                    <div class="metric-value">{spread_data.get('half_life_days', 0):.1f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col5:
            is_coint = spread_data.get('is_cointegrated', False)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Cointegrated</div>
                    <div class="metric-value">{'✅' if is_coint else '❌'}</div>
                </div>
            """, unsafe_allow_html=True)

        # -------------------- SIGNAL BOX --------------------
        signal = spread_data.get('signal', 'NEUTRAL')
        signal_class = f"signal-{signal}"
        signal_text = {
            "LONG_SPREAD": "🔺 BUY SPREAD (Go Long T1, Short T2)",
            "SHORT_SPREAD": "🔻 SHORT SPREAD (Short T1, Go Long T2)",
            "NEUTRAL": "⚪ NEUTRAL (No Trade)"
        }
        st.markdown(f"""
            <div class="signal-box {signal_class}">
                {signal_text.get(signal, signal)}
            </div>
        """, unsafe_allow_html=True)

        # -------------------- GAUGE VISUALIZATION --------------------
        st.markdown("### 📊 Current Z-Score Gauge")

        z_score = spread_data.get('current_zscore', 0)

        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=z_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Z-Score", 'font': {'size': 24}},
            delta={'reference': 0, 'relative': False},
            gauge={
                'axis': {'range': [-3.5, 3.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E3A8A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-3.5, -entry_zscore], 'color': '#D1FAE5'},
                    {'range': [-entry_zscore, -exit_zscore], 'color': '#FEF3C7'},
                    {'range': [-exit_zscore, exit_zscore], 'color': '#F3F4F6'},
                    {'range': [exit_zscore, entry_zscore], 'color': '#FEF3C7'},
                    {'range': [entry_zscore, 3.5], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': z_score
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add entry/exit labels below gauge
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"**Entry Short:** +{entry_zscore}")
        with col2:
            st.markdown(f"**Exit Short:** +{exit_zscore}")
        with col3:
            st.markdown("**Mean:** 0")
        with col4:
            st.markdown(f"**Exit Long:** -{exit_zscore}")
        with col5:
            st.markdown(f"**Entry Long:** -{entry_zscore}")

        # -------------------- TRADE LOGIC EXPLANATION --------------------
        with st.expander("📖 Understanding the Signals"):
            is_coint = spread_data.get('is_cointegrated', False)
            pvalue = spread_data.get('adf_pvalue', 1.0)

            st.markdown(f"""
                **Data Summary:**
                - **Data Points:** {spread_data.get('data_points', 0)} trading days
                - **ADF Test p-value:** {pvalue:.4f}
                - **Cointegrated:** {'✅ Yes' if is_coint else '❌ No'}
                - **Half-Life:** {spread_data.get('half_life_days', 'N/A'):.1f} days
                - **Hedge Ratio:** {spread_data.get('hedge_ratio', 'N/A'):.4f}
                - **Current Z-Score:** {spread_data.get('current_zscore', 'N/A'):.3f}

                ---

                **How this works:**

                1. **Cointegration**: The two stocks move together over time. If they drift apart, they will eventually converge.
                2. **Z-Score**: Measures how many standard deviations the current spread is from its historical mean.
                3. **Entry Signal**:
                   - `LONG_SPREAD`: Z-score < -{entry_zscore} → Buy the spread (expect it to increase)
                   - `SHORT_SPREAD`: Z-score > {entry_zscore} → Sell the spread (expect it to decrease)
                4. **Exit Signal**: Z-score returns to within ±{exit_zscore} → Close the trade

                **For the Naspers/Prosus pair:**
                - These are twin stocks. Naspers holds ~43% of Prosus, so they should trade in a fixed ratio.
                - Any deviation is a potential arbitrage opportunity.

                **Signal Colors:**
                - 🟢 **Green Zone (LONG)**: Z-score < -{entry_zscore} → Buy opportunity
                - 🟡 **Yellow Zone**: Z-score between -{entry_zscore} and {entry_zscore} → Wait/Exit
                - 🔴 **Red Zone (SHORT)**: Z-score > {entry_zscore} → Sell opportunity
            """)

        # -------------------- RAW DATA DISPLAY --------------------
        with st.expander("🔍 Raw API Response"):
            st.json(spread_data)

    else:
        st.warning("No data available. Check your FastAPI connection.")

# ==================== BACKTEST TAB ====================
with tab_backtest:
    st.markdown("### 🧪 Walk-Forward Backtest")
    st.caption(
        "Rolling hedge-ratio re-estimation (no lookahead), rolling z-score entry/exit, "
        "transaction costs and slippage applied on both legs."
    )

    with st.form("backtest_form"):
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_lookback_days = st.number_input(
                "History to Fetch (days)", value=1500, step=100, min_value=100,
            )
            bt_coint_method = st.selectbox("Cointegration Method", ["engle_granger", "johansen"])
            bt_capital = st.number_input("Starting Capital", value=100_000.0, step=10_000.0)
        with bt_col2:
            bt_lookback_window = st.number_input(
                "Re-fit Lookback Window (days)", value=252, step=21, min_value=30,
            )
            bt_reestimate_every = st.number_input(
                "Re-fit Every (days)", value=21, step=7, min_value=1,
            )
            bt_zscore_window = st.number_input(
                "Z-Score Window (days)", value=30, step=5, min_value=5,
            )
        with bt_col3:
            bt_transaction_cost_bps = st.number_input("Transaction Cost (bps/leg)", value=5.0, step=1.0)
            bt_slippage_bps = st.number_input("Slippage (bps/leg)", value=2.0, step=1.0)
            bt_require_coint = st.checkbox("Require cointegration to trade", value=True)

        run_clicked = st.form_submit_button("▶️ Run Backtest")

    if run_clicked:
        backtest_request = {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "lookback_days": int(bt_lookback_days),
            "coint_method": bt_coint_method,
            "entry_zscore": entry_zscore,
            "exit_zscore": exit_zscore,
            "lookback_window": int(bt_lookback_window),
            "reestimate_every": int(bt_reestimate_every),
            "zscore_window": int(bt_zscore_window),
            "require_cointegration": bt_require_coint,
            "transaction_cost_bps": bt_transaction_cost_bps,
            "slippage_bps": bt_slippage_bps,
            "capital": bt_capital,
        }
        with st.spinner("Running walk-forward backtest..."):
            response = run_backtest(backtest_request)
        # Only replace a prior successful result once we have a new one --
        # a failed re-run (bad ticker, timeout, backend hiccup) shouldn't
        # wipe out the last good backtest still on screen.
        if response is not None:
            st.session_state["backtest_result"] = {"request": backtest_request, "response": response}

    backtest_state = st.session_state.get("backtest_result")

    if backtest_state:
        backtest_result = backtest_state["response"]
        # Use the params that actually produced this result, not whatever
        # is currently sitting in the sidebar -- the user may have changed
        # the ticker/z-score inputs since the last "Run Backtest" click.
        result_ticker1 = backtest_result["ticker1"]
        result_ticker2 = backtest_result["ticker2"]
        result_entry_zscore = backtest_state["request"]["entry_zscore"]
        metrics = backtest_result["metrics"]

        st.markdown(f"**Results for:** `{result_ticker1}` vs `{result_ticker2}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{metrics['total_return']:.2%}")
        m2.metric("CAGR", f"{metrics['cagr']:.2%}")
        m3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        m4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Sortino", f"{metrics['sortino']:.2f}")
        m6.metric("Win Rate", f"{metrics['win_rate']:.2%}")
        m7.metric("Avg Trade P&L", f"{metrics['avg_trade_pnl']:.2f}")
        m8.metric("Number of Trades", f"{int(metrics['num_trades'])}")

        equity_df = pd.DataFrame(backtest_result["equity_curve"])
        equity_df["date"] = pd.to_datetime(equity_df["date"])

        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=equity_df["date"], y=equity_df["equity"], name="Equity",
            line=dict(color="#1E3A8A"),
        ))
        equity_fig.update_layout(
            title="Equity Curve", height=350, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(equity_fig, use_container_width=True)

        zscore_df = pd.DataFrame(backtest_result["zscore_series"])
        zscore_df["date"] = pd.to_datetime(zscore_df["date"])

        zscore_fig = go.Figure()
        zscore_fig.add_trace(go.Scatter(
            x=zscore_df["date"], y=zscore_df["zscore"], name="Z-Score",
            line=dict(color="#7C3AED"),
        ))
        zscore_fig.add_hline(y=result_entry_zscore, line_dash="dash", line_color="#EF4444")
        zscore_fig.add_hline(y=-result_entry_zscore, line_dash="dash", line_color="#10B981")
        zscore_fig.add_hline(y=0, line_color="#9CA3AF")
        zscore_fig.update_layout(
            title="Rolling Z-Score", height=300, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(zscore_fig, use_container_width=True)

        st.markdown("#### Trade Log")
        trades = backtest_result["trades"]
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        else:
            st.info("No trades were generated over this period.")
    else:
        st.info("Configure parameters above and click **Run Backtest**.")

# -------------------- AUTO-REFRESH LOGIC --------------------
if auto_refresh:
    time.sleep(5)
    st.rerun()

# -------------------- FOOTER --------------------
st.markdown("""
    <div class="footer">
        Built with ❤️ by Mathematics and Applied Mathematics Graduate |
        Data from Yahoo Finance |
        <a href="https://github.com" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)
