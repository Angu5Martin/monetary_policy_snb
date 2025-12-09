
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class AssetSideAnalyzer:
    
    def __init__(self):
        self.fx_investments = None
        self.gold_reserves = None
        self.equity_provisions = None
        self.fx_transactions = None
        self.asset_side_interventions = None
        
    def load_asset_data(self, processed_data: Dict[str, pd.DataFrame]) -> None:
        
        # Load core components
        self.fx_investments = processed_data['foreign_currency_investments'].copy()
        self.gold_reserves = processed_data['gold_reserves'].copy()
        self.equity_provisions = processed_data['equity_provisions'].copy()
        self.fx_transactions = processed_data['fx_transactions'].copy()
    
    def create_combined_asset_dataset(self) -> pd.DataFrame:
    
        if self.fx_investments is None:
            raise ValueError("Data not loaded")
                
        combined_df = self.fx_investments[['Date', 'Foreign Currency Investments (CHF Millions)']].copy()
        combined_df.rename(columns={
            'Foreign Currency Investments (CHF Millions)': 'FX_Investments'
        }, inplace=True)
        
        # Merge gold reserves
        gold_clean = self.gold_reserves[['Date', 'Gold Reserves (CHF Millions)']].copy()
        gold_clean.rename(columns={
            'Gold Reserves (CHF Millions)': 'Gold_Reserves'
        }, inplace=True)
        combined_df = combined_df.merge(gold_clean, on='Date', how='left')
        
        # Merge equity provisions
        equity_clean = self.equity_provisions[['Date', 'Equity Provisions (CHF Millions)']].copy()
        equity_clean.rename(columns={
            'Equity Provisions (CHF Millions)': 'Equity_Provisions'
        }, inplace=True)
        combined_df = combined_df.merge(equity_clean, on='Date', how='left')
        
        # Note: FX transactions are now merged ONLY after quarterly aggregation
        for col in ['Gold_Reserves', 'Equity_Provisions']:
            combined_df[col] = combined_df[col].fillna(method='ffill').fillna(0)
        
        # Sort by date
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        return combined_df
    
    def calculate_asset_side_interventions(self, combined_df: pd.DataFrame) -> pd.DataFrame:

        df = combined_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
                
        # Create quarterly aggregation
        df['Quarter'] = df['Date'].dt.to_period('Q')
        df['Quarter_Start'] = df['Quarter'].dt.start_time
        
        quarterly_levels = df.groupby('Quarter').agg({
            'Date': 'last',
            'FX_Investments': 'last',
            'Gold_Reserves': 'last', 
            'Equity_Provisions': 'last'
        }).reset_index()
        
        quarterly_levels['Quarter_Start'] = quarterly_levels['Quarter'].dt.start_time
        
        # Count how many months exist in each quarter to identify incomplete latest quarter
        month_counts = df.groupby('Quarter')['Date'].nunique().reset_index().rename(columns={'Date': 'Month_Count'})
        quarterly_levels = quarterly_levels.merge(month_counts, on='Quarter', how='left')

        # Calculate quarter-over-quarter changes for all components
        quarterly_levels['FX_Investments_Change'] = quarterly_levels['FX_Investments'].diff()
        quarterly_levels['Gold_Reserves_Change'] = quarterly_levels['Gold_Reserves'].diff()
        quarterly_levels['Equity_Provisions_Change'] = quarterly_levels['Equity_Provisions'].diff()
        
        # Calculate asset-side intervention proxy
        quarterly_levels['Asset_Side_Intervention'] = (
            quarterly_levels['FX_Investments_Change'] + 
            quarterly_levels['Gold_Reserves_Change'] - 
            quarterly_levels['Equity_Provisions_Change']
        )
        
        quarterly_levels['Asset_Side_FX_Only'] = quarterly_levels['FX_Investments_Change']
        quarterly_levels['Asset_Side_FX_Gold'] = (quarterly_levels['FX_Investments_Change'] + 
                                                 quarterly_levels['Gold_Reserves_Change'])
        
        # Only exclude the latest quarter if incomplete (<3 months of data).
        if not quarterly_levels.empty:
            latest_q = quarterly_levels['Quarter'].max()
            latest_months = quarterly_levels.loc[quarterly_levels['Quarter'] == latest_q, 'Month_Count'].iloc[0]
            if pd.notna(latest_months) and latest_months < 3:
                quarterly_levels = quarterly_levels[quarterly_levels['Quarter'] != latest_q].copy()

        # Merge FX transactions quarterly AFTER computing changes and completeness filtering
        fx_q = self.fx_transactions.copy()
        fx_q['Date'] = pd.to_datetime(fx_q['Date'])
        fx_q['Quarter'] = fx_q['Date'].dt.to_period('Q')
        fx_quarter = fx_q.groupby('Quarter').agg({'Total': 'sum'}).reset_index().rename(columns={'Total': 'Reported_FX_Transactions'})
        quarterly_levels = quarterly_levels.merge(fx_quarter, on='Quarter', how='left')
        # Remove first row (no change calculation possible)
        quarterly_levels = quarterly_levels.iloc[1:].copy()
        
        self.asset_side_interventions = quarterly_levels
        return quarterly_levels

    def calculate_asset_side_monthly_interventions(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly intervention proxy without merging reported FX transactions.

        Returns a DataFrame with monthly changes analogous to the quarterly proxy:
        Asset_Side_Intervention_Monthly = ΔFX_Investments + ΔGold_Reserves - ΔEquity_Provisions
        Also includes FX-only and FX+Gold variants for symmetry.
        """
        df = combined_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df['FX_Investments_Change'] = df['FX_Investments'].diff()
        df['Gold_Reserves_Change'] = df['Gold_Reserves'].diff()
        df['Equity_Provisions_Change'] = df['Equity_Provisions'].diff()

        df['Asset_Side_Intervention_Monthly'] = (
            df['FX_Investments_Change'] + df['Gold_Reserves_Change'] - df['Equity_Provisions_Change']
        )
        df['Asset_Side_FX_Only_Monthly'] = df['FX_Investments_Change']
        df['Asset_Side_FX_Gold_Monthly'] = df['FX_Investments_Change'] + df['Gold_Reserves_Change']

        # Drop first row (no prior month for diff)
        df = df.iloc[1:].copy()
        return df
    
    def compare_with_reported_transactions(self, df: pd.DataFrame) -> pd.DataFrame:

        comparison_df = df[['Quarter_Start', 'Asset_Side_Intervention', 'Asset_Side_FX_Only', 
                   'Asset_Side_FX_Gold', 'Reported_FX_Transactions']].copy()
        
        comparison_df = comparison_df.rename(columns={'Quarter_Start': 'Date'})
        
        # Calculate differences
        comparison_df['Difference_Full'] = (
            comparison_df['Asset_Side_Intervention'] - comparison_df['Reported_FX_Transactions']
        )
        comparison_df['Difference_FX_Only'] = (
            comparison_df['Asset_Side_FX_Only'] - comparison_df['Reported_FX_Transactions']
        )
        comparison_df['Difference_FX_Gold'] = (
            comparison_df['Asset_Side_FX_Gold'] - comparison_df['Reported_FX_Transactions']
        )
        
        # Calculate correlations 
        comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
        clean_data = comparison_df.dropna()
        clean_data_2020 = clean_data[clean_data['Date'] >= '2020-01-01']
        
        corr_full_2020 = clean_data_2020['Asset_Side_Intervention'].corr(clean_data_2020['Reported_FX_Transactions'])
        corr_fx_only_2020 = clean_data_2020['Asset_Side_FX_Only'].corr(clean_data_2020['Reported_FX_Transactions'])
        corr_fx_gold_2020 = clean_data_2020['Asset_Side_FX_Gold'].corr(clean_data_2020['Reported_FX_Transactions'])
            
        # Store correlations for later use
        comparison_df['Correlation_Full'] = corr_full_2020
        comparison_df['Correlation_FX_Only'] = corr_fx_only_2020
        comparison_df['Correlation_FX_Gold'] = corr_fx_gold_2020

        return comparison_df
    
    def create_comparison_visualization(self, comparison_df: pd.DataFrame) -> None:
        colors = {
            'reported': '#D62728',
            'predicted': '#063871',
            'trend': '#88BDF7',
            'perfect': '#9DAFC6',
        }

        import matplotlib.ticker as mtick

        plot_df = comparison_df.copy()
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
        plot_df = plot_df[plot_df['Date'] >= pd.to_datetime('2020-01-01')].copy()
        plot_df['Quarter'] = plot_df['Date'].dt.to_period('Q')
        plot_df['QLabel'] = plot_df['Quarter'].astype(str)

        def format_quarter_axis(ax, labels):
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, axis='y', alpha=0.25)

        # 1) Full proxy vs Reported
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        x = range(len(plot_df))
        ax1.plot(x, plot_df['Asset_Side_Intervention'], label='Asset-Side Proxy (Full)', color=colors['predicted'], linewidth=2.5)
        ax1.plot(x, plot_df['Reported_FX_Transactions'], label='Reported FX Transactions', color=colors['reported'], linewidth=2.5, linestyle='--')
        ax1.axhline(0, color='black', alpha=0.3)
        ax1.set_ylabel('Quarterly Flow (CHF Millions)')
        format_quarter_axis(ax1, plot_df['QLabel'].tolist())
        for spine in ['top','right']:
            ax1.spines[spine].set_visible(False)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        ax1.grid(False)
        plt.tight_layout(); plt.show()

        # 2) FX-only vs Reported
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.plot(x, plot_df['Asset_Side_FX_Only'], label='FX Reserves Only', color=colors['predicted'], linewidth=2.5)
        ax2.plot(x, plot_df['Reported_FX_Transactions'], label='Reported FX Transactions', color=colors['reported'], linewidth=2.5, linestyle='--')
        ax2.axhline(0, color='black', alpha=0.3)
        ax2.set_ylabel('Quarterly Flow (CHF Millions)')
        format_quarter_axis(ax2, plot_df['QLabel'].tolist())
        for spine in ['top','right']:
            ax2.spines[spine].set_visible(False)
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        ax2.grid(False)
        plt.tight_layout(); plt.show()

        # 3) Correlation scatter
        clean = plot_df[['Asset_Side_Intervention', 'Reported_FX_Transactions']].dropna()
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.scatter(clean['Asset_Side_Intervention'], clean['Reported_FX_Transactions'], color=colors['predicted'], alpha=0.85, s=80, edgecolors='black', linewidth=0.8)
        if len(clean) > 2:
            corr = clean['Asset_Side_Intervention'].corr(clean['Reported_FX_Transactions'])
            min_val = min(clean.min())
            max_val = max(clean.max())
            ax3.plot([min_val, max_val], [min_val, max_val], color=colors['perfect'], linestyle='--', linewidth=2, label='Perfect Correlation')
            z = np.polyfit(clean['Asset_Side_Intervention'], clean['Reported_FX_Transactions'], 1)
            p = np.poly1d(z)
            ax3.plot(clean['Asset_Side_Intervention'], p(clean['Asset_Side_Intervention']), color=colors['trend'], linewidth=2, label='Trend')
        ax3.set_xlabel('Asset-Side Proxy (CHF Millions)')
        ax3.set_ylabel('Reported FX Transactions (CHF Millions)')
        for spine in ['top','right']:
            ax3.spines[spine].set_visible(False)
        ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax3.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax3.grid(False)
        plt.tight_layout(); plt.show()


def create_asset_side_analyzer() -> AssetSideAnalyzer:

    return AssetSideAnalyzer()


def run_full_asset_side_analysis(processed_data: Dict[str, pd.DataFrame], monthly: bool = False) -> pd.DataFrame:
    """Run asset-side analysis.

    Args:
        processed_data: Dict of processed SNB datasets.
        monthly: If True, return monthly proxy DataFrame without reported FX transactions.
                 If False, perform quarterly analysis and visualize.

    Returns:
        Monthly proxy DataFrame if monthly=True else quarterly comparison DataFrame.
    """
    analyzer = create_asset_side_analyzer()
    analyzer.load_asset_data(processed_data)
    combined_df = analyzer.create_combined_asset_dataset()
    if monthly:
        return analyzer.calculate_asset_side_monthly_interventions(combined_df)
    intervention_df = analyzer.calculate_asset_side_interventions(combined_df)
    comparison_df = analyzer.compare_with_reported_transactions(intervention_df)
    analyzer.create_comparison_visualization(comparison_df)

