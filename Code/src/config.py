"""
Configuration file for SNB Analysis Pipeline.

This module contains all configuration parameters, constants, and mappings
used throughout the analysis pipeline.
"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

BASE_URL = "https://data.snb.ch/api/cube/{cube}/data/csv/{lang}"

# SNB Data Cube IDs
CUBE_BALANCE_SHEET_ITEMS = "snbbipo"
CUBE_FX_TRANSACTIONS = "snbfxtr"
CUBE_FX_INVESTMENTS = "snbcurrc"
CUBE_FX_RATES = "devkum"
CUBE_CHF_INDEX = "devwkieffim"
CUBE_SARON_RATES = "zimoma"
CUBE_IMF_RESERVES = "sddsilv7m"

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Rolling window for calculating weights (in quarters for FX investments data)
ROLLING_WINDOW_QUARTERS = 4  # 1 year lookback

# Currencies to analyze (matching the FX investments data)
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'Other']

# Currency mapping for FX rates
FX_RATE_CURRENCY_MAP = {
    'USD': 'USD',
    'EUR': 'EUR', 
    'JPY': 'JPY/100',  # Note: JPY is quoted per 100 units
    'GBP': 'GBP',
    'CAD': 'CAD',
    'Other': 'CHF_INDEX'  # Use CHF effective exchange rate index for "Other" currencies
}

# FX investments cube mapping
FX_CURRENCY_MAPPING = {
    'ICHF0': 'USD',
    'ICHF1': 'EUR', 
    'ICHF2': 'JPY',
    'ICHF3': 'GBP',
    'ICHF4': 'CAD',
    'ICHF5': 'Other'
}

# =============================================================================
# PORTFOLIO DECOMPOSITION
# =============================================================================

# Portfolio allocation mapping for SNB_reserve_decomposition
PORTFOLIO_TYPES = ['Equity', 'Bond']

# Regional equity index mappings
EQUITY_REGIONS = {
    'USD': 'S&P 500',      # US equity market
    'EUR': 'STOXX Europe 600', # European equity market
    'JPY': 'Nikkei 225',   # Japanese equity market
    'GBP': 'FTSE 100',     # UK equity market
    'CAD': 'S&P TSX',      # Canadian equity market
    'Other': 'MSCI World'  # Global equity proxy for other currencies
}

# =============================================================================
# DATA PATHS
# =============================================================================

# Base directories
DATA_DIR = "data"
EQUITY_DATA_DIR = "data/equity_prices"
OUTPUT_DIR = "outputs"
NOTEBOOKS_DIR = "notebooks"

# Output file names
FX_IMPACT_FILE = "fx_impact_analysis.csv"
FX_WEIGHTS_FILE = "fx_weights_analysis.csv"
MONTHLY_HOLDINGS_FILE = "monthly_holdings_by_currency.csv"
QUARTERLY_COMPARISON_FILE = "quarterly_intervention_comparison.csv"

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Color palette for currencies
CURRENCY_COLORS = {
    'USD': '#1f77b4',
    'EUR': '#ff7f0e', 
    'JPY': '#2ca02c',
    'GBP': '#d62728',
    'CAD': '#9467bd',
    'Other': '#8c564b'
}

# Figure settings
FIGURE_SIZE = (15, 10)
DPI = 100
FONT_SIZE = 12

# =============================================================================
# API PARAMETERS
# =============================================================================

# Dimension selections for specific cubes
CUBE_DIMENSION_SELECTIONS = {
    CUBE_FX_RATES: "D0(M1),D1(EUR1,GBP1,USD1,CAD1,JPY100)",
    CUBE_BALANCE_SHEET_ITEMS: "D0(D,GB,N,VRGSF,ES,VB,GBI,GD,FRGSF,VF,GFG,RE)",
    CUBE_CHF_INDEX: "D0(N),D1(G)",
    CUBE_FX_INVESTMENTS: "D0(ICHF0,ICHF1,ICHF2,ICHF3,ICHF4,ICHF5)",
    CUBE_IMF_RESERVES: "D0(RAXG)"
}

# Request timeout settings
REQUEST_TIMEOUT = 60

# =============================================================================
# DATA VALIDATION
# =============================================================================

# Expected data frequency
EXPECTED_FREQUENCIES = {
    CUBE_BALANCE_SHEET_ITEMS: 'M',  # Monthly
    CUBE_FX_TRANSACTIONS: 'Q',      # Quarterly  
    CUBE_FX_INVESTMENTS: 'Q',       # Quarterly
    CUBE_FX_RATES: 'M',             # Monthly
    CUBE_CHF_INDEX: 'M',             # Monthly
    CUBE_SARON_RATES: 'M',           # Monthly
    CUBE_IMF_RESERVES: 'M'          # Monthly
}

# Minimum data quality thresholds
MIN_DATA_POINTS = 12  # Minimum 1 year of data
MIN_CORRELATION_THRESHOLD = 0.5  # For validation comparisons
