"""
TICKER UNIVERSE - JSE PAIRS TRADING
Curated, sector-grouped list of JSE-listed tickers (plus a couple of
known cross-asset pairs) for the cointegration screener.

Why curated rather than the full JSE All-Share / Top 40:
- Sector peers are far more likely to be cointegrated than random pairs,
  so grouping by sector keeps the screener fast and the results sensible.
- A small, named, editable list is easier to verify and re-run reliably
  against yfinance than scraping a live constituent list each run.
- Ticker symbols on JSE can shift (renames, delistings) -- review this
  list periodically and update as needed.

Tickers verified against Yahoo Finance JSE listings (.JO suffix) as of
the last update to this file. If a download fails for a ticker, the
screener will log a warning and continue with the rest of the universe.
"""

# -------------------- SECTOR GROUPS --------------------
# Pairs are only tested *within* a sector by default (see pair_screener.py),
# since cross-sector pairs are rarely cointegrated and testing all
# combinations across the full universe would be wasteful.

BANKS = [
    "FSR.JO",   # FirstRand
    "SBK.JO",   # Standard Bank Group
    "ABG.JO",   # Absa Group
    "NED.JO",   # Nedbank Group
    "CPI.JO",   # Capitec Bank Holdings
]

MINING = [
    "AGL.JO",   # Anglo American
    "BHP.JO",   # BHP Group (JSE-listed line)
    "ANG.JO",   # AngloGold Ashanti
    "GFI.JO",   # Gold Fields
    "IMP.JO",   # Impala Platinum (Implats)
    "EXX.JO",   # Exxaro Resources
]

TELECOMS = [
    "MTN.JO",   # MTN Group
    "VOD.JO",   # Vodacom Group
]

RETAIL = [
    "SHP.JO",   # Shoprite Holdings
    "WHL.JO",   # Woolworths Holdings
    "MRP.JO",   # Mr Price Group
]

INSURANCE = [
    "SLM.JO",   # Sanlam
    "OMU.JO",   # Old Mutual
    "DSY.JO",   # Discovery
]

# -------------------- STRUCTURAL / CROSS-ASSET PAIRS --------------------
# Pairs with a known economic link rather than just sector membership.
# These are tested individually rather than as part of a sector group.

STRUCTURAL_PAIRS = [
    ("NPN.JO", "PRX.JO"),  # Naspers / Prosus - Naspers holds ~43% of Prosus
    ("SOL.JO", "CL=F"),    # Sasol / WTI Crude - Sasol's earnings are oil-linked
]

# -------------------- COMBINED UNIVERSE --------------------
SECTOR_GROUPS = {
    "Banks": BANKS,
    "Mining": MINING,
    "Telecoms": TELECOMS,
    "Retail": RETAIL,
    "Insurance": INSURANCE,
}

ALL_TICKERS = sorted(set(
    t for group in SECTOR_GROUPS.values() for t in group
) | set(t for pair in STRUCTURAL_PAIRS for t in pair))
