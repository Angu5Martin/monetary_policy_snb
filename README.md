# Swiss National Bank FX Intervention Analysis

Real-time prediction and analysis of Swiss National Bank (SNB) foreign exchange market interventions using balance sheet data.

## Overview

This project develops three complementary approaches to predict SNB foreign exchange interventions before official quarterly reports are published. By analyzing monthly balance sheet data, we can estimate intervention activity with a 1-week to 1-month reporting lag, compared to the SNB's quarterly disclosure schedule.

### Key Features

- **Three Prediction Models**:
  - **Asset-Side Approach (1-month delay)**: Tracks changes in foreign currency investments, gold reserves, and equity provisions
  - **Liability-Side Approach (1-month delay)**: Monitors comprehensive sight deposit adjustments across multiple balance sheet components
  - **Valuation-Adjusted Approach (1-week delay)**: Adjusts reserve changes for currency and equity market valuation effects

- **Automated Data Collection**: Fetches latest SNB balance sheet data and reported FX transactions from official sources

- **Comprehensive Visualization**: 
  - Quarterly time series with Credit Suisse emergency lending period highlighted
  - Correlation analysis (full period vs. excluding crisis period)
  - 24-month rolling intervention predictions
  - Directional indicators for buying/selling foreign currency

## Project Structure

```
Code/
├── src/
│   ├── analysis/           # Three intervention prediction models
│   │   ├── asset_side_first.py
│   │   ├── sight_deposits.py
│   │   └── asset_side_second.py
│   ├── data/              # Data collection and preprocessing
│   │   ├── snb_collector.py
│   │   └── preprocessor.py
│   └── utils/             # Helper functions
│       └── helpers.py
├── notebooks/             # Analysis workflow
│   └── snb_step_by_step_analysis.ipynb
└── data/                  # Raw data storage
    ├── bond_data/
    └── equity_prices/
```

## Installation

### Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- requests

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Angu5Martin/monetary_policy_snb.git
cd monetary_policy_snb
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib requests
```

## Usage

### Quick Start

Run the complete analysis pipeline:

```python
from src.data.snb_collector import collect_all_snb_data
from src.data.preprocessor import preprocess_all_snb_data
from src.analysis.asset_side_first import run_full_asset_side_analysis
from src.analysis.sight_deposits import run_full_liability_side_analysis
from src.analysis.asset_side_second import run_full_asset_side_second_analysis

# Collect data from 2010 onwards
raw_data = collect_all_snb_data(from_date="2010-01")
processed_data = preprocess_all_snb_data(raw_data)

# Run all three models
run_full_asset_side_analysis(processed_data)
run_full_liability_side_analysis(processed_data)
run_full_asset_side_second_analysis(processed_data)
```

### Monthly Mode

Get monthly intervention predictions without visualization:

```python
# Returns DataFrame with monthly predictions
monthly_asset = run_full_asset_side_analysis(processed_data, monthly=True)
monthly_liability = run_full_liability_side_analysis(processed_data, monthly=True)
monthly_valuation = run_full_asset_side_second_analysis(processed_data, monthly=True)
```

## Methodology

### Asset-Side Approach
Predicts interventions by tracking changes in:
- Foreign currency investments (main driver)
- Gold reserves
- Equity provisions (valuation adjustments)

**Formula**: ΔFX Investments + ΔGold - ΔEquity Provisions

### Liability-Side Approach
Captures intervention effects through comprehensive sight deposit adjustments:
- Sight deposit changes
- Repo transaction adjustments
- Confederation account movements
- FX liability changes
- Secured loans and debt certificates

### Valuation-Adjusted Approach
Directly estimates interventions by removing market valuation effects from reserve changes using:
- Currency-specific FX rate movements
- Equity index performance
- Bond return indexes

## Data Sources

- **SNB Balance Sheet Data**: [Swiss National Bank - Statistical Data Portal](https://data.snb.ch/)
- **FX Transaction Reports**: SNB Quarterly Reports
- **Exchange Rates**: SNB official rates
- **Equity Indices**: MSCI World, S&P 500, STOXX Europe 600, etc.

## Contributing

This project was developed for the Monetary Policy Seminar at UZH. Contributions, issues, and feature requests are welcome.

---

*Last Updated: December 2025*
