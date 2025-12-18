import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SightDepositsAnalyzer:
    
    def __init__(self):
        self.sight_deposits = None
        self.bank_notes = None
        self.repo_liabilities = None
        self.debt_certificates = None
        self.foreign_bank_sight_deposits = None
        self.liability_based_interventions = None
        self.claims_chf_repo = None
        self.secured_loans = None
        self.fx_liabilities = None

    def load_liability_data(self, processed_data: Dict[str, pd.DataFrame]) -> None:
        
        # Load core components
        self.sight_deposits = processed_data['bank_sight_deposits'].copy()
        self.bank_notes = processed_data['bank_notes_in_circulation'].copy()
        self.repo_liabilities = processed_data['liabilities_chf_repo'].copy()
        self.debt_certificates = processed_data['snb_debt_certificates'].copy()
        self.foreign_bank_sight_deposits = processed_data['foreign_bank_sight_deposits'].copy()
        self.claims_chf_repo = processed_data['claims_chf_repo'].copy()
        self.secured_loans = processed_data['secured_loans'].copy()
        self.fx_liabilities = processed_data['fx_liabilities'].copy()
        self.fx_transactions = processed_data['fx_transactions'].copy()
        self.confederation_amount = processed_data['confederation_amount'].copy()
    
    def create_combined_liability_dataset(self) -> pd.DataFrame:

        if self.sight_deposits is None:
            raise ValueError("Data not loaded.")
                
        # Start with sight deposits as the base
        combined_df = self.sight_deposits.copy()
        combined_df.rename(columns={
            'Bank Sight Deposits (CHF Millions)': 'Sight_Deposits'
        }, inplace=True)
        
        # Merge other components
        # Bank notes
        bank_notes_clean = self.bank_notes[['Date', 'Bank Notes in Circulation (CHF Millions)']].copy()
        bank_notes_clean.rename(columns={
            'Bank Notes in Circulation (CHF Millions)': 'Bank_Notes'
        }, inplace=True)
        combined_df = combined_df.merge(bank_notes_clean, on='Date', how='left')
        
        # Repo liabilities
        repo_clean = self.repo_liabilities[['Date', 'Liabilities from CHF Repo Transactions (CHF Millions)']].copy()
        repo_clean.rename(columns={
            'Liabilities from CHF Repo Transactions (CHF Millions)': 'Repo_Liabilities'
        }, inplace=True)
        combined_df = combined_df.merge(repo_clean, on='Date', how='left')
        
        # Debt certificates
        debt_clean = self.debt_certificates[['Date', 'SNB Debt Certificates (CHF Millions)']].copy()
        debt_clean.rename(columns={
            'SNB Debt Certificates (CHF Millions)': 'Debt_Certificates'
        }, inplace=True)
        combined_df = combined_df.merge(debt_clean, on='Date', how='left')

        # Foreign Bank Sight Deposits (GBI)
        foreign_bank_sight_deposits_clean = self.foreign_bank_sight_deposits[['Date', 'Amount from Foreign Bank Sight Deposits (CHF Millions)']].copy()
        foreign_bank_sight_deposits_clean.rename(columns={
            'Amount from Foreign Bank Sight Deposits (CHF Millions)': 'Foreign_Bank_Sight_Deposits'
        }, inplace=True)
        combined_df = combined_df.merge(foreign_bank_sight_deposits_clean, on='Date', how='left')

        # Claims from CHF Repo Transactions
        claims_chf_repo_clean = self.claims_chf_repo[['Date', 'Claims from CHF Repo Transactions (CHF Millions)']].copy()
        claims_chf_repo_clean.rename(columns={
            'Claims from CHF Repo Transactions (CHF Millions)': 'Claims_from_CHF_Repo'
        }, inplace=True)
        combined_df = combined_df.merge(claims_chf_repo_clean, on='Date', how='left')

        # Secured Loans
        secured_loans_clean = self.secured_loans[['Date', 'Secured Loans (CHF Millions)']].copy()
        secured_loans_clean.rename(columns={
            'Secured Loans (CHF Millions)': 'Secured_Loans'
        }, inplace=True)
        combined_df = combined_df.merge(secured_loans_clean, on='Date', how='left')

        # FX Liabilities
        fx_liabilities_clean = self.fx_liabilities[['Date', 'FX Liabilities (CHF Millions)']].copy()
        fx_liabilities_clean.rename(columns={
            'FX Liabilities (CHF Millions)': 'FX_Liabilities'
        }, inplace=True)
        combined_df = combined_df.merge(fx_liabilities_clean, on='Date', how='left')

        # Confederation Amount
        confed_clean = self.confederation_amount[['Date', 'Amount from the Confederation (CHF Millions)']].copy()
        confed_clean.rename(columns={
            'Amount from the Confederation (CHF Millions)': 'Confederation_Amount'
        }, inplace=True)
        combined_df = combined_df.merge(confed_clean, on='Date', how='left')

        # FX transactions will be merged only after quarterly aggregation now
        
        # Fill forward level series to avoid spurious NaNs at quarter boundaries
        for c in [
            "Bank_Notes",
            "Repo_Liabilities",
            "Debt_Certificates",
            "Foreign_Bank_Sight_Deposits",
            "Claims_from_CHF_Repo",
            "Secured_Loans",
            "FX_Liabilities",
            "Confederation_Amount",
            "Sight_Deposits",
        ]:
            if c in combined_df.columns:
                combined_df[c] = combined_df[c].ffill()

        # Sort by date
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        print(f"Combined dataset created with {len(combined_df)} observations")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        return combined_df
    
    def calculate_liability_based_interventions(self, combined_df: pd.DataFrame) -> pd.DataFrame:

        df = combined_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        # Quarterly aggregation to match asset-side methodology
        df['Quarter'] = df['Date'].dt.to_period('Q')

        agg_spec = {
            'Date': 'last',
            'Sight_Deposits': 'last',
            'Bank_Notes': 'last',
            'Repo_Liabilities': 'last',
            'Debt_Certificates': 'last',
            'Foreign_Bank_Sight_Deposits': 'last',
            'Claims_from_CHF_Repo': 'last',
            'Secured_Loans': 'last',
            'FX_Liabilities': 'last',
            'Confederation_Amount': 'last'
        }

        quarterly_levels = df.groupby('Quarter').agg(agg_spec).reset_index()
        quarterly_levels['Quarter_Start'] = quarterly_levels['Quarter'].dt.start_time
        # Count months per quarter to detect incomplete latest quarter
        month_counts = df.groupby('Quarter')['Date'].nunique().reset_index().rename(columns={'Date': 'Month_Count'})
        quarterly_levels = quarterly_levels.merge(month_counts, on='Quarter', how='left')

        # Quarter-over-quarter changes
        quarterly_levels['Sight_Deposits_Change'] = quarterly_levels['Sight_Deposits'].diff()
        quarterly_levels['Repo_Liabilities_Change'] = quarterly_levels['Repo_Liabilities'].diff()
        quarterly_levels['Debt_Certificates_Change'] = quarterly_levels['Debt_Certificates'].diff()
        quarterly_levels['foreign_bank_sight_deposits_change'] = quarterly_levels['Foreign_Bank_Sight_Deposits'].diff()
        quarterly_levels['Bank_Notes_Adjusted'] = quarterly_levels['Bank_Notes'].diff()
        quarterly_levels['Claims_from_CHF_Repo_Change'] = quarterly_levels['Claims_from_CHF_Repo'].diff()
        quarterly_levels['Secured_Loans_Change'] = quarterly_levels['Secured_Loans'].diff()
        # quarterly_levels['FX_Liabilities_Change'] = quarterly_levels['FX_Liabilities'].diff().fillna(0)
        quarterly_levels['Confederation_Amount_Change'] = quarterly_levels['Confederation_Amount'].diff()

        # Proxies
        quarterly_levels['Intervention_Proxy_Simple'] = quarterly_levels['Sight_Deposits_Change']
        quarterly_levels['Intervention_Proxy_Comprehensive'] = (
            quarterly_levels['Sight_Deposits_Change']
            - quarterly_levels['Claims_from_CHF_Repo_Change']
            - quarterly_levels['Secured_Loans_Change']
            - quarterly_levels['Bank_Notes_Adjusted']
            + quarterly_levels['Confederation_Amount_Change']
            + quarterly_levels['Repo_Liabilities_Change']
            + quarterly_levels['Debt_Certificates_Change']
            # + quarterly_levels['FX_Liabilities_Change']
            + quarterly_levels['foreign_bank_sight_deposits_change']
        )

        quarterly_levels['Liability_Based_Intervention'] = quarterly_levels['Intervention_Proxy_Comprehensive']

        # Only exclude the latest quarter if incomplete (<3 months of data).
        if not quarterly_levels.empty:
            latest_q = quarterly_levels['Quarter'].max()
            latest_months = quarterly_levels.loc[quarterly_levels['Quarter'] == latest_q, 'Month_Count'].iloc[0]
            if pd.notna(latest_months) and latest_months < 3:
                quarterly_levels = quarterly_levels[quarterly_levels['Quarter'] != latest_q].copy()

        # Align naming with asset-side for comparison step
        # Merge FX transactions quarterly AFTER computing changes and completeness filtering
        fx_q = self.fx_transactions.copy()
        fx_q['Date'] = pd.to_datetime(fx_q['Date'])
        fx_q['Quarter'] = fx_q['Date'].dt.to_period('Q')
        fx_quarter = fx_q.groupby('Quarter').agg({'Total':'sum'}).reset_index().rename(columns={'Total':'Reported_FX_Transactions'})
        quarterly_levels = quarterly_levels.merge(fx_quarter, on='Quarter', how='left')

        # Drop first row (no change)
        quarterly_levels = quarterly_levels.iloc[1:].copy()

        self.liability_based_interventions = quarterly_levels
        return quarterly_levels

    def calculate_monthly_liability_based_interventions(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly liability-side proxies without merging reported FX transactions.

        Returns a DataFrame with monthly changes analogous to quarterly comprehensive/simple proxies.
        Intervention_Proxy_Simple_Monthly = ΔSight_Deposits
        Intervention_Proxy_Comprehensive_Monthly mirrors quarterly comprehensive formula using monthly diffs.
        """
        df = combined_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Monthly changes
        df['Sight_Deposits_Change'] = df['Sight_Deposits'].diff()
        df['Repo_Liabilities_Change'] = df['Repo_Liabilities'].diff()
        df['Debt_Certificates_Change'] = df['Debt_Certificates'].diff()
        df['foreign_bank_sight_deposits_change'] = df['Foreign_Bank_Sight_Deposits'].diff()
        df['Bank_Notes_Adjusted'] = df['Bank_Notes'].diff()
        df['Claims_from_CHF_Repo_Change'] = df['Claims_from_CHF_Repo'].diff()
        df['Secured_Loans_Change'] = df['Secured_Loans'].diff()
        # df['FX_Liabilities_Change'] = df['FX_Liabilities'].diff().fillna(0)
        df['Confederation_Amount_Change'] = df['Confederation_Amount'].diff()

        # Monthly proxies
        df['Intervention_Proxy_Simple_Monthly'] = df['Sight_Deposits_Change']
        df['Intervention_Proxy_Comprehensive_Monthly'] = (
            df['Sight_Deposits_Change']
            - df['Claims_from_CHF_Repo_Change']
            - df['Secured_Loans_Change']
            - df['Bank_Notes_Adjusted']
            + df['Confederation_Amount_Change']
            + df['Repo_Liabilities_Change']
            + df['Debt_Certificates_Change']
            # + df['FX_Liabilities_Change']
            + df['foreign_bank_sight_deposits_change']
        )

        df = df.iloc[1:].copy()  # drop first diff NA row
        return df
    
    
    def compare_with_fx_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect quarterly df with 'Quarter_Start' and 'Reported_FX_Transactions'
        cols_needed = ['Quarter_Start', 'Intervention_Proxy_Comprehensive', 'Intervention_Proxy_Simple']
        avail_cols = [c for c in cols_needed if c in df.columns]

        base_cols = avail_cols + (['Reported_FX_Transactions'] if 'Reported_FX_Transactions' in df.columns else [])
        comparison_df = df[base_cols].copy()
        comparison_df = comparison_df.rename(columns={'Quarter_Start': 'Date'})

        # Differences (optional diagnostics)
        if 'Intervention_Proxy_Comprehensive' in comparison_df.columns and 'Reported_FX_Transactions' in comparison_df.columns:
            comparison_df['Difference_Comprehensive'] = (
                comparison_df['Intervention_Proxy_Comprehensive'] - comparison_df['Reported_FX_Transactions']
            )
        if 'Intervention_Proxy_Simple' in comparison_df.columns and 'Reported_FX_Transactions' in comparison_df.columns:
            comparison_df['Difference_Simple'] = (
                comparison_df['Intervention_Proxy_Simple'] - comparison_df['Reported_FX_Transactions']
            )

        # Correlations for 2020+
        comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
        clean = comparison_df.dropna()
        clean_2020 = clean[clean['Date'] >= '2020-01-01']

        if 'Reported_FX_Transactions' in clean_2020.columns and 'Intervention_Proxy_Comprehensive' in clean_2020.columns and len(clean_2020) > 2:
            corr_comp = clean_2020['Intervention_Proxy_Comprehensive'].corr(clean_2020['Reported_FX_Transactions'])
            comparison_df['Correlation_Comprehensive'] = corr_comp
        if 'Reported_FX_Transactions' in clean_2020.columns and 'Intervention_Proxy_Simple' in clean_2020.columns and len(clean_2020) > 2:
            corr_simple = clean_2020['Intervention_Proxy_Simple'].corr(clean_2020['Reported_FX_Transactions'])
            comparison_df['Correlation_Simple'] = corr_simple

        return comparison_df

    def create_comparison_visualization(self, comparison_df: pd.DataFrame) -> None:
        colors = {
            'reported': '#D62728',
            'predicted': '#063871',
            'trend': '#88BDF7',
            'perfect': '#9DAFC6'
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
            ax.grid(True, axis='y', alpha=0.3)

        x = range(len(plot_df))

        # 1) Comprehensive proxy vs reported with 2023Q1-Q2 shading
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        # Add shading for 2023Q1-Q2
        q1_2023 = pd.Period('2023Q1', freq='Q')
        q2_2023 = pd.Period('2023Q2', freq='Q')
        shade_mask = (plot_df['Quarter'] >= q1_2023) & (plot_df['Quarter'] <= q2_2023)
        if shade_mask.any():
            shade_idx = plot_df[shade_mask].index
            shade_start = list(plot_df.index).index(shade_idx[0])
            shade_end = list(plot_df.index).index(shade_idx[-1]) + 1
            ax1.axvspan(shade_start - 0.5, shade_end - 0.5, color='lightgrey', alpha=0.3, zorder=0)
        if 'Intervention_Proxy_Comprehensive' in plot_df.columns:
            ax1.plot(x, plot_df['Intervention_Proxy_Comprehensive'], label='Predicted Intervention', color=colors['predicted'], linewidth=2.5)
        ax1.plot(x, plot_df['Reported_FX_Transactions'], label='Reported FX Transactions', color=colors['reported'], linewidth=2.5, linestyle='--')
        ax1.axhline(0, color='black', alpha=0.3)
        ax1.set_ylabel('Quarterly Flow (CHF Millions)', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        format_quarter_axis(ax1, plot_df['QLabel'].tolist())
        for spine in ['top','right']:
            ax1.spines[spine].set_visible(False)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
        ax1.grid(False)
        # Add label to shaded region after plot is created
        if shade_mask.any():
            mid_shade = (shade_start + shade_end - 1) / 2
            y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
            y_pos = ax1.get_ylim()[1] - 0.01 * y_range
            ax1.text(mid_shade, y_pos, 'Credit Suisse\nEmergency Lending', 
                    ha='center', va='top', fontsize=9, color='dimgray', style='italic')
        # Add directional arrows on right side inside plot
        x_arrow = len(plot_df) - 1.5
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min
        arrow_y_up = y_max * 0.6 if y_max > 0 else y_range * 0.2
        arrow_y_down = y_min * 0.6 if y_min < 0 else -y_range * 0.2
        # Up arrow (green)
        arrow_up_start = arrow_y_up - y_range * 0.08
        ax1.annotate('', xy=(x_arrow, arrow_y_up), xytext=(x_arrow, arrow_up_start),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
        arrow_up_center = (arrow_y_up + arrow_up_start) / 2
        ax1.text(x_arrow - 0.5, arrow_up_center, 'Buying FC\nSelling CHF', 
                fontsize=8, color='green', va='center', ha='right', weight='bold')
        # Down arrow (red)
        arrow_down_start = arrow_y_down + y_range * 0.08
        ax1.annotate('', xy=(x_arrow, arrow_y_down), xytext=(x_arrow, arrow_down_start),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
        arrow_down_center = (arrow_y_down + arrow_down_start) / 2
        ax1.text(x_arrow - 0.5, arrow_down_center, 'Selling FC\nBuying CHF', 
                fontsize=8, color='red', va='center', ha='right', weight='bold')
        plt.tight_layout(); plt.show()

        # Statistics: full period vs excluding 2023Q1-Q2
        if 'Intervention_Proxy_Comprehensive' in plot_df.columns:
            clean_full = plot_df[['Intervention_Proxy_Comprehensive', 'Reported_FX_Transactions']].dropna()
            clean_excl = plot_df[~shade_mask][['Intervention_Proxy_Comprehensive', 'Reported_FX_Transactions']].dropna()
            if len(clean_full) > 2:
                corr_full = clean_full['Intervention_Proxy_Comprehensive'].corr(clean_full['Reported_FX_Transactions'])
                mae_full = abs(clean_full['Intervention_Proxy_Comprehensive'] - clean_full['Reported_FX_Transactions']).mean()
                rmse_full = np.sqrt(((clean_full['Intervention_Proxy_Comprehensive'] - clean_full['Reported_FX_Transactions'])**2).mean())
                print('\nLiability-Side Proxy Summary (Full Period 2020+):')
                print(f'  Correlation: {corr_full:.3f}')
                print(f'  Mean Absolute Error: {mae_full:,.1f} CHF Millions')
                print(f'  Root Mean Square Error: {rmse_full:,.1f} CHF Millions')
                print(f'  Quarterly data points: {len(clean_full)}')
            if len(clean_excl) > 2:
                corr_excl = clean_excl['Intervention_Proxy_Comprehensive'].corr(clean_excl['Reported_FX_Transactions'])
                mae_excl = abs(clean_excl['Intervention_Proxy_Comprehensive'] - clean_excl['Reported_FX_Transactions']).mean()
                rmse_excl = np.sqrt(((clean_excl['Intervention_Proxy_Comprehensive'] - clean_excl['Reported_FX_Transactions'])**2).mean())
                print('\nLiability-Side Proxy Summary (Excluding 2023Q1-Q2):')
                print(f'  Correlation: {corr_excl:.3f}')
                print(f'  Mean Absolute Error: {mae_excl:,.1f} CHF Millions')
                print(f'  Root Mean Square Error: {rmse_excl:,.1f} CHF Millions')
                print(f'  Quarterly data points: {len(clean_excl)}')

        # 2) Monthly bar chart (past 24 months)
        monthly_df = self.calculate_monthly_liability_based_interventions(self.combined_df)
        monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=24)
        recent = monthly_df[monthly_df['Date'] >= cutoff].copy()
        recent = recent.sort_values('Date')
        recent['MonthLabel'] = recent['Date'].dt.strftime('%Y-%m')
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        xm = range(len(recent))
        if 'Intervention_Proxy_Comprehensive_Monthly' in recent.columns:
            ax2.bar(xm, recent['Intervention_Proxy_Comprehensive_Monthly'], color=colors['predicted'], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        tick_pos = [i for i in xm if i % 3 == 0]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([recent['MonthLabel'].iloc[i] for i in tick_pos], rotation=45, ha='right', fontsize=11)
        ax2.set_ylabel('Monthly Flow (CHF Millions)', fontsize=14)
        ax2.tick_params(axis='y', which='major', labelsize=12)
        for spine in ['top','right']:
            ax2.spines[spine].set_visible(False)
        ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax2.grid(False)
        plt.tight_layout(); plt.show()

        # 3) Correlation scatter - Full Period
        if 'Intervention_Proxy_Comprehensive' in plot_df.columns:
            clean_full = plot_df[['Intervention_Proxy_Comprehensive', 'Reported_FX_Transactions']].dropna()
        else:
            clean_full = pd.DataFrame(columns=['Intervention_Proxy_Comprehensive','Reported_FX_Transactions'])
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        if len(clean_full) > 2:
            ax3.scatter(clean_full['Intervention_Proxy_Comprehensive'], clean_full['Reported_FX_Transactions'], color=colors['predicted'], alpha=0.85, s=80, edgecolors='black', linewidth=0.8)
            corr_full = clean_full['Intervention_Proxy_Comprehensive'].corr(clean_full['Reported_FX_Transactions'])
            min_val = min(clean_full.min())
            max_val = max(clean_full.max())
            ax3.plot([min_val, max_val], [min_val, max_val], color=colors['perfect'], linestyle='--', linewidth=2, label='Perfect Correlation')
            z = np.polyfit(clean_full['Intervention_Proxy_Comprehensive'], clean_full['Reported_FX_Transactions'], 1)
            p = np.poly1d(z)
            ax3.plot(clean_full['Intervention_Proxy_Comprehensive'], p(clean_full['Intervention_Proxy_Comprehensive']), color=colors['trend'], linewidth=2, label=f'Trend (r={corr_full:.3f})')
        ax3.set_xlabel('Predicted Intervention (CHF Millions)', fontsize=14)
        ax3.set_ylabel('Reported FX Transactions (CHF Millions)', fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        for spine in ['top','right']:
            ax3.spines[spine].set_visible(False)
        ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax3.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
        ax3.grid(False)
        plt.tight_layout(); plt.show()

        # 4) Correlation scatter - Excluding 2023Q1-Q2
        q1_2023 = pd.Period('2023Q1', freq='Q')
        q2_2023 = pd.Period('2023Q2', freq='Q')
        shade_mask = (plot_df['Quarter'] >= q1_2023) & (plot_df['Quarter'] <= q2_2023)
        if 'Intervention_Proxy_Comprehensive' in plot_df.columns:
            clean_excl = plot_df[~shade_mask][['Intervention_Proxy_Comprehensive', 'Reported_FX_Transactions']].dropna()
        else:
            clean_excl = pd.DataFrame(columns=['Intervention_Proxy_Comprehensive','Reported_FX_Transactions'])
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        if len(clean_excl) > 2:
            ax4.scatter(clean_excl['Intervention_Proxy_Comprehensive'], clean_excl['Reported_FX_Transactions'], color=colors['predicted'], alpha=0.85, s=80, edgecolors='black', linewidth=0.8)
            corr_excl = clean_excl['Intervention_Proxy_Comprehensive'].corr(clean_excl['Reported_FX_Transactions'])
            min_val = min(clean_excl.min())
            max_val = max(clean_excl.max())
            ax4.plot([min_val, max_val], [min_val, max_val], color=colors['perfect'], linestyle='--', linewidth=2, label='Perfect Correlation')
            z = np.polyfit(clean_excl['Intervention_Proxy_Comprehensive'], clean_excl['Reported_FX_Transactions'], 1)
            p = np.poly1d(z)
            ax4.plot(clean_excl['Intervention_Proxy_Comprehensive'], p(clean_excl['Intervention_Proxy_Comprehensive']), color=colors['trend'], linewidth=2, label=f'Trend (r={corr_excl:.3f})')
        ax4.set_xlabel('Predicted Intervention (CHF Millions)', fontsize=14)
        ax4.set_ylabel('Reported FX Transactions (CHF Millions)', fontsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        for spine in ['top','right']:
            ax4.spines[spine].set_visible(False)
        ax4.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax4.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
        ax4.grid(False)
        plt.tight_layout(); plt.show()


def create_sight_deposits_analyzer() -> SightDepositsAnalyzer:
    """
    Factory function to create a SightDepositsAnalyzer instance.
    
    Returns:
        Configured SightDepositsAnalyzer instance
    """
    return SightDepositsAnalyzer()


def run_full_liability_side_analysis(processed_data: Dict[str, pd.DataFrame], monthly: bool = False) -> pd.DataFrame:
    """Run liability-side analysis with optional monthly proxy mode.

    Args:
        processed_data: Dict of processed SNB datasets.
        monthly: If True, return monthly liability-side proxy DataFrame without reported FX transactions.
                 If False, perform quarterly analysis and visualize against reported FX transactions.

    Returns:
        Monthly proxy DataFrame if monthly=True else quarterly comparison DataFrame.
    """
    analyzer = create_sight_deposits_analyzer()
    analyzer.load_liability_data(processed_data)
    combined_df = analyzer.create_combined_liability_dataset()
    analyzer.combined_df = combined_df  # Store for visualization
    if monthly:
        return analyzer.calculate_monthly_liability_based_interventions(combined_df)
    intervention_df = analyzer.calculate_liability_based_interventions(combined_df)
    comparison_df = analyzer.compare_with_fx_transactions(intervention_df)
    analyzer.create_comparison_visualization(comparison_df)


