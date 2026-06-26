"""
STREAMLIT DASHBOARD - JSE PAIRS TRADING
North West University - Quant Project
Connects to FastAPI backend for live spread monitoring
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
#api_url = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")
api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="https://live-pairs-trading-dashboard.onrender.com"  # Your deployed API URL
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
        st.error("❌ Cannot connect to FastAPI. Make sure it's running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# -------------------- MAIN DASHBOARD --------------------
st.markdown('<div class="main-header">🇿🇦 JSE Pairs Trading Dashboard</div>', unsafe_allow_html=True)
st.markdown(f"**Pair:** `{ticker1}` vs `{ticker2}` | **Lookback:** {lookback} days")

# Fetch current spread data
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