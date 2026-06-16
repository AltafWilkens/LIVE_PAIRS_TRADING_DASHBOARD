jse-pairs-trading/
├── README.md                 # Full LaTeX math explanation
├── requirements.txt          # All Python dependencies
├── Dockerfile                # Containerization
├── main.py                   # FastAPI backend (starter script above)
├── models/
│   ├── kalman_filter.py      # Dynamic hedge ratio
│   └── cointegration.py      # ADF / Johansen tests
├── services/
│   ├── data_fetcher.py       # yfinance wrapper
│   └── signal_generator.py   # Entry/exit logic
├── tests/
│   ├── test_cointegration.py # Pytest unit tests
│   └── test_kalman.py
├── dashboard/
│   └── app.py                # Streamlit frontend
└── backtests/
    └── backtest_engine.py    # Historical P&L simulation