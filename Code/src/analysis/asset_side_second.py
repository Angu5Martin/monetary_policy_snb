
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import warnings
import logging

warnings.filterwarnings('ignore')


class AssetSideSecondAnalyzer:
	def __init__(self):
		self.imf_reserves = None
		self.fx_transactions = None
		self.fx_investments = None  
		self.monthly_fx_investments = None 
		self.fx_rates = None  
		self.equity_prices = None  
		self.bond_return_indexes = None  
		self.monthly_weights = None
		self.monthly_returns = None
		self.monthly_valuation_effects = None
		self.monthly_interventions = None
		self.quarterly_comparison = None

	def load_prediction_data(self, processed_data: Dict[str, pd.DataFrame]) -> None:
		self.imf_reserves = processed_data['imf_reserves'].copy()
		self.fx_transactions = processed_data['fx_transactions'].copy()
		self.fx_investments = processed_data['fx_investments'].copy() 
		# Monthly balance sheet FX investments (level) for monthly comparison plot
		if 'foreign_currency_investments' in processed_data:
			self.monthly_fx_investments = processed_data['foreign_currency_investments'].copy()
		self.fx_rates = processed_data['fx_rates'].copy()
		# Normalize FX column names (e.g., JPY/100 -> JPY) for downstream alignment with weights
		fx_cols = [c for c in self.fx_rates.columns if c != 'Date']
		if 'JPY/100' in fx_cols and 'JPY' not in fx_cols:
			self.fx_rates.rename(columns={'JPY/100': 'JPY'}, inplace=True)

	def _project_root(self) -> Path:
		p = Path(__file__).resolve()
		candidates = [p.parents[2], p.parents[3]]  
		for cand in candidates:
			data_dir = cand / 'data'
			if data_dir.exists() and (data_dir / 'equity_prices').exists():
				return cand
		return p.parents[2]

	def load_market_data(self) -> None:
		base_root = self._project_root()
		base_data = base_root / 'data'
		equity_dir = base_data / 'equity_prices'
		bond_dir = base_data / 'bond_data'

		equity_frames = []
		equity_mapping = {
			'S&P 500': 'USD',
			'STOXX Europe 600': 'EUR',
			'Nikkei 225 Stock Average': 'JPY',
			'FTSE 100': 'GBP',
			'S&P TSX Composite': 'CAD',
			'MSCI World Index': 'WORLD'
		}
		for f in equity_dir.glob('*.csv'):
			try:
				df = pd.read_csv(f)
				# Expect a Date column and a Price/Value column; take first two
				df.columns = [c.strip() for c in df.columns]
				date_col = [c for c in df.columns if 'Date' in c or 'date' in c][0]
				value_col = [c for c in df.columns if c != date_col][0]
				df = df[[date_col, value_col]].copy()
				df.rename(columns={date_col: 'Date', value_col: 'Price'}, inplace=True)
				df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
				# Map filename to currency code
				base_name = f.name.split('(')[0].strip()
				for k, v in equity_mapping.items():
					if k in base_name:
						df['Currency'] = v
						break
				equity_frames.append(df)
			except Exception as e:
				continue
		if equity_frames:
			equity_df = pd.concat(equity_frames, ignore_index=True)
			# Pivot to wide price table
			self.equity_prices = equity_df.pivot_table(index='Date', columns='Currency', values='Price', aggfunc='last').reset_index()
			self.equity_prices = self.equity_prices.sort_values('Date').reset_index(drop=True)
		else:
			self.equity_prices = pd.DataFrame(columns=['Date'])

		bond_frames = []
		bond_mapping = {
			'US_return_index.xlsx': 'USD',
			'Euro_return_index.xlsx': 'EUR',
			'Japan_return_index.xlsx': 'JPY',
			'UK_return_index.xlsx': 'GBP',
			'Canada_return_index.xlsx': 'CAD'
		}
		for f in bond_dir.glob('*.xlsx'):
			if f.name not in bond_mapping:
				continue
			try:
				df = pd.read_excel(f)
				df.columns = [c.strip() for c in df.columns]
				date_col = [c for c in df.columns if 'Date' in c or 'date' in c][0]
				value_col = [c for c in df.columns if c != date_col][0]
				df = df[[date_col, value_col]].copy()
				df.rename(columns={date_col: 'Date', value_col: 'Index'}, inplace=True)
				df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
				df['Currency'] = bond_mapping[f.name]
				bond_frames.append(df)
			except Exception as e:
				continue
		if bond_frames:
			bond_df = pd.concat(bond_frames, ignore_index=True)
			self.bond_return_indexes = bond_df.pivot_table(index='Date', columns='Currency', values='Index', aggfunc='last').reset_index()
			self.bond_return_indexes = self.bond_return_indexes.sort_values('Date').reset_index(drop=True)
		else:
			self.bond_return_indexes = pd.DataFrame(columns=['Date'])

	def compute_weights(self) -> None:
		df = self.fx_investments.copy()
		# Date already quarterly period end (YYYY-MM). Convert to datetime then to month string again for consistency.
		df['Date'] = pd.to_datetime(df['Date'])
		currency_cols = [c for c in df.columns if c != 'Date']
		df['Total'] = df[currency_cols].sum(axis=1)
		weight_df = df[['Date'] + currency_cols].copy()
		for c in currency_cols:
			weight_df[c] = np.where(df['Total'] == 0, 0, df[c] / df['Total'])
		weight_df['Date'] = weight_df['Date'].dt.strftime('%Y-%m')
		# Expand to monthly range covering IMF reserves
		monthly_dates = sorted(self.imf_reserves['Date'].unique())
		expanded = pd.DataFrame({'Date': monthly_dates})
		weights_monthly = expanded.merge(weight_df, on='Date', how='left')
		weights_monthly = weights_monthly.ffill()
		self.monthly_weights = weights_monthly

	def compute_monthly_returns(self) -> None:
		# Equity returns
		eq = self.equity_prices.copy()
		bd = self.bond_return_indexes.copy()
		fx = self.fx_rates.copy()
		for frame in [eq, bd, fx]:
			if frame.empty:
				continue
			frame['Date'] = pd.to_datetime(frame['Date']).dt.strftime('%Y-%m')

		# Compute pct change for each asset class
		def pct_change_wide(frame: pd.DataFrame, label: str) -> pd.DataFrame:
			if frame.empty:
				return pd.DataFrame(columns=['Date'])
			out = frame.copy()
			cols = [c for c in out.columns if c != 'Date']
			for c in cols:
				# Robust numeric conversion: remove thousands separators, stray spaces
				series_str = out[c].astype(str).str.replace(',', '').str.replace(' ', '')
				out[c] = pd.to_numeric(series_str, errors='coerce')
			
			for c in cols:
				out[c] = out[c].pct_change()
			out.rename(columns={c: f'{c}_{label}' for c in cols}, inplace=True)
			return out

		eq_ret = pct_change_wide(eq, 'EQUITY')
		bd_ret = pct_change_wide(bd, 'BOND')
		fx_ret = pct_change_wide(fx, 'FX')

		# Merge returns together
		merged = eq_ret.merge(bd_ret, on='Date', how='outer').merge(fx_ret, on='Date', how='outer').sort_values('Date').reset_index(drop=True)
		self.monthly_returns = merged

	def estimate_valuation_effects(self) -> None:
		if self.monthly_returns is None or self.monthly_weights is None:
			raise ValueError('Returns or weights not computed')
		returns = self.monthly_returns.copy()
		weights = self.monthly_weights.copy()
		combined = weights.merge(returns, on='Date', how='left')

		currency_list = [c for c in weights.columns if c != 'Date']
		valuation_rows = []
		for _, row in combined.iterrows():
			date = row['Date']
			total_effect = 0.0
			components = {}
			for cur in currency_list:
				w = row.get(cur, np.nan)
				if pd.isna(w) or w == 0:
					continue
				eq_ret = row.get(f'{cur}_EQUITY', 0.0)
				bd_ret = row.get(f'{cur}_BOND', 0.0)
				fx_ret = row.get(f'{cur}_FX', 0.0)
				local_asset_ret = 0.75 * bd_ret + 0.25 * eq_ret
				approx_total_ret = local_asset_ret + fx_ret
				effect = w * approx_total_ret
				components[f'Valuation_{cur}'] = effect
				total_effect += effect
			valuation_rows.append({'Date': date, 'Valuation_Effect': total_effect, **components})
		self.monthly_valuation_effects = pd.DataFrame(valuation_rows).sort_values('Date').reset_index(drop=True)

	def compute_interventions(self) -> None:
		reserves = self.imf_reserves.copy()
		reserves['Date'] = pd.to_datetime(reserves['Date']).dt.strftime('%Y-%m')
		reserves['IMF_Reserves_Change'] = reserves['IMF_Reserves'].diff()
		val = self.monthly_valuation_effects.copy()
		df = reserves.merge(val, on='Date', how='left')
		# Multiply valuation effect (fractional) by previous reserves level to get CHF amount
		df['Prev_Reserves'] = df['IMF_Reserves'].shift(1)
		df['Valuation_Amount'] = df['Valuation_Effect'] * df['Prev_Reserves']
		df['Predicted_Intervention'] = df['IMF_Reserves_Change'] - df['Valuation_Amount']
		self.monthly_interventions = df

	def aggregate_quarterly(self) -> None:
		# Build quarterly sums from monthly interventions
		df = self.monthly_interventions.copy()
		df['Date_dt'] = pd.to_datetime(df['Date'])
		df['Quarter'] = df['Date_dt'].dt.to_period('Q')
		month_counts = df.groupby('Quarter')['Date_dt'].nunique().reset_index().rename(columns={'Date_dt':'Month_Count'})
		q_agg = df.groupby('Quarter').agg({
			'Predicted_Intervention': 'sum',
			'Valuation_Amount': 'sum',
			'IMF_Reserves_Change': 'sum'
		}).reset_index().merge(month_counts, on='Quarter', how='left')
		q_agg['Quarter_Start'] = q_agg['Quarter'].dt.start_time
		# Quarterly reported FX transactions
		fx_trx = self.fx_transactions.copy()
		fx_trx['Date'] = pd.to_datetime(fx_trx['Date'])
		fx_trx['Quarter'] = fx_trx['Date'].dt.to_period('Q')
		fx_quarter = fx_trx.groupby('Quarter').agg({'Total': 'sum'}).reset_index().rename(columns={'Total': 'Reported_FX_Transactions'})
		# Determine latest interventions quarter and completeness
		latest_int_q = q_agg['Quarter'].max()
		latest_int_months = q_agg.loc[q_agg['Quarter'] == latest_int_q, 'Month_Count'].iloc[0]
		# OPTION 1 implementation: include all completed predicted quarters regardless of reported horizon.
		# Only exclude the latest quarter if incomplete (<3 months of data).
		mask = pd.Series(True, index=q_agg.index)
		if latest_int_months < 3:
			mask &= q_agg['Quarter'] != latest_int_q
		q_agg = q_agg[mask].copy()
		comparison = q_agg.merge(fx_quarter, on='Quarter', how='left')
		comparison = comparison.sort_values('Quarter').reset_index(drop=True)
		# Drop first quarter if changes present (diff base)
		if len(comparison) > 1 and comparison[['Predicted_Intervention','Valuation_Amount','IMF_Reserves_Change']].iloc[0].notna().any():
			comparison = comparison.iloc[1:].copy()
		self.quarterly_comparison = comparison

	def compare_with_reported_transactions(self) -> pd.DataFrame:
		df = self.quarterly_comparison.copy()
		df['Difference_Predicted'] = df['Predicted_Intervention'] - df['Reported_FX_Transactions']
		df['Difference_Valuation'] = df['Valuation_Amount'] - df['Reported_FX_Transactions']
		# Correlation (2020+)
		df['Quarter_Start'] = pd.to_datetime(df['Quarter_Start'])
		clean = df[df['Quarter_Start'] >= pd.to_datetime('2020-01-01')].dropna(subset=['Predicted_Intervention','Reported_FX_Transactions'])
		if len(clean) > 2:
			corr_pred = clean['Predicted_Intervention'].corr(clean['Reported_FX_Transactions'])
		else:
			corr_pred = np.nan
		df['Correlation_Predicted'] = corr_pred
		return df.rename(columns={'Quarter_Start': 'Date'})

	def create_comparison_visualization(self, comparison_df: pd.DataFrame) -> None:
		# Simplified presentation: (1) Predicted vs Reported (quarterly), (2) Correlation (quarterly), (3) Monthly IMF vs FX Investments
		colors = {
			'reported': '#D62728',
			'predicted': '#063871',
			'trend': '#88BDF7',
			'perfect': '#9DAFC6',
		}
		# Quarterly base
		q_df = comparison_df.copy()
		q_df['Date'] = pd.to_datetime(q_df['Date'])
		q_df = q_df[q_df['Date'] >= pd.to_datetime('2020-01-01')].copy()
		q_df['Quarter'] = q_df['Date'].dt.to_period('Q')
		q_df['QLabel'] = q_df['Quarter'].astype(str)
		xq = range(len(q_df))

		def format_quarter_axis(ax, labels):
			ax.set_xticks(range(len(labels)))
			ax.set_xticklabels(labels, rotation=45, ha='right')
			ax.grid(True, axis='y', alpha=0.3)

		# 1) Predicted vs Reported with 2023Q1-Q2 shading
		fig1, ax1 = plt.subplots(figsize=(12,8))
		# Add shading for 2023Q1-Q2
		q1_2023 = pd.Period('2023Q1', freq='Q')
		q2_2023 = pd.Period('2023Q2', freq='Q')
		shade_mask = (q_df['Quarter'] >= q1_2023) & (q_df['Quarter'] <= q2_2023)
		if shade_mask.any():
			shade_idx = q_df[shade_mask].index
			shade_start = list(q_df.index).index(shade_idx[0])
			shade_end = list(q_df.index).index(shade_idx[-1]) + 1
			ax1.axvspan(shade_start - 0.5, shade_end - 0.5, color='lightgrey', alpha=0.3, zorder=0)
		ax1.plot(xq, q_df['Predicted_Intervention'], label='Predicted Intervention', color=colors['predicted'], linewidth=2.5)
		ax1.plot(xq, q_df['Reported_FX_Transactions'], label='Reported FX Transactions', color=colors['reported'], linewidth=2.5, linestyle='--')
		ax1.axhline(0, color='black', alpha=0.3)
		ax1.set_ylabel('Quarterly Flow (CHF Millions)', fontsize=14)
		ax1.tick_params(axis='both', which='major', labelsize=12)
		format_quarter_axis(ax1, q_df['QLabel'].tolist())
		for spine in ['top','right']:
			ax1.spines[spine].set_visible(False)
		import matplotlib.ticker as mtick
		ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax1.grid(False)
		ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
		# Add label to shaded region after plot is created
		if shade_mask.any():
			mid_shade = (shade_start + shade_end - 1) / 2
			y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
			y_pos = ax1.get_ylim()[1] - 0.01 * y_range
			ax1.text(mid_shade, y_pos, 'Credit Suisse\nEmergency Lending', 
					ha='center', va='top', fontsize=9, color='dimgray', style='italic')
		# Add directional arrows on right side inside plot
		x_arrow = len(q_df) - 1.5
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
		clean_full = q_df[['Predicted_Intervention', 'Reported_FX_Transactions']].dropna()
		clean_excl = q_df[~shade_mask][['Predicted_Intervention', 'Reported_FX_Transactions']].dropna()
		if len(clean_full) > 2:
			corr_full = clean_full['Predicted_Intervention'].corr(clean_full['Reported_FX_Transactions'])
			mae_full = abs(clean_full['Predicted_Intervention'] - clean_full['Reported_FX_Transactions']).mean()
			rmse_full = np.sqrt(((clean_full['Predicted_Intervention'] - clean_full['Reported_FX_Transactions'])**2).mean())
			print('\nAsset-Side Second Proxy Summary (Full Period 2020+):')
			print(f'  Correlation: {corr_full:.3f}')
			print(f'  Mean Absolute Error: {mae_full:,.1f} CHF Millions')
			print(f'  Root Mean Square Error: {rmse_full:,.1f} CHF Millions')
			print(f'  Quarterly data points: {len(clean_full)}')
		if len(clean_excl) > 2:
			corr_excl = clean_excl['Predicted_Intervention'].corr(clean_excl['Reported_FX_Transactions'])
			mae_excl = abs(clean_excl['Predicted_Intervention'] - clean_excl['Reported_FX_Transactions']).mean()
			rmse_excl = np.sqrt(((clean_excl['Predicted_Intervention'] - clean_excl['Reported_FX_Transactions'])**2).mean())
			print('\nAsset-Side Second Proxy Summary (Excluding 2023Q1-Q2):')
			print(f'  Correlation: {corr_excl:.3f}')
			print(f'  Mean Absolute Error: {mae_excl:,.1f} CHF Millions')
			print(f'  Root Mean Square Error: {rmse_excl:,.1f} CHF Millions')
			print(f'  Quarterly data points: {len(clean_excl)}')

		# 2) Correlation scatter - Full Period
		clean_full = q_df[['Predicted_Intervention','Reported_FX_Transactions']].dropna()
		fig2, ax2 = plt.subplots(figsize=(12,8))
		if len(clean_full) > 2:
			corr_full = clean_full['Predicted_Intervention'].corr(clean_full['Reported_FX_Transactions'])
			min_val = min(clean_full.min())
			max_val = max(clean_full.max())
			ax2.scatter(clean_full['Predicted_Intervention'], clean_full['Reported_FX_Transactions'], color=colors['predicted'], alpha=0.85, s=80, edgecolors='black', linewidth=0.8)
			ax2.plot([min_val, max_val],[min_val, max_val], color=colors['perfect'], linestyle='--', linewidth=2, label='Perfect Correlation')
			z = np.polyfit(clean_full['Predicted_Intervention'], clean_full['Reported_FX_Transactions'], 1)
			p = np.poly1d(z)
			ax2.plot(clean_full['Predicted_Intervention'], p(clean_full['Predicted_Intervention']), color=colors['trend'], linewidth=2, label=f'Trend (r={corr_full:.3f})')
		ax2.set_xlabel('Predicted Intervention (CHF Millions)', fontsize=14)
		ax2.set_ylabel('Reported FX Transactions (CHF Millions)', fontsize=14)
		ax2.tick_params(axis='both', which='major', labelsize=12)
		for spine in ['top','right']:
			ax2.spines[spine].set_visible(False)
		ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax2.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
		ax2.grid(False)
		plt.tight_layout(); plt.show()

		# 2b) Correlation scatter - Excluding 2023Q1-Q2
		q1_2023 = pd.Period('2023Q1', freq='Q')
		q2_2023 = pd.Period('2023Q2', freq='Q')
		shade_mask = (q_df['Quarter'] >= q1_2023) & (q_df['Quarter'] <= q2_2023)
		clean_excl = q_df[~shade_mask][['Predicted_Intervention','Reported_FX_Transactions']].dropna()
		fig2b, ax2b = plt.subplots(figsize=(12,8))
		if len(clean_excl) > 2:
			corr_excl = clean_excl['Predicted_Intervention'].corr(clean_excl['Reported_FX_Transactions'])
			min_val = min(clean_excl.min())
			max_val = max(clean_excl.max())
			ax2b.scatter(clean_excl['Predicted_Intervention'], clean_excl['Reported_FX_Transactions'], color=colors['predicted'], alpha=0.85, s=80, edgecolors='black', linewidth=0.8)
			ax2b.plot([min_val, max_val],[min_val, max_val], color=colors['perfect'], linestyle='--', linewidth=2, label='Perfect Correlation')
			z = np.polyfit(clean_excl['Predicted_Intervention'], clean_excl['Reported_FX_Transactions'], 1)
			p = np.poly1d(z)
			ax2b.plot(clean_excl['Predicted_Intervention'], p(clean_excl['Predicted_Intervention']), color=colors['trend'], linewidth=2, label=f'Trend (r={corr_excl:.3f})')
		ax2b.set_xlabel('Predicted Intervention (CHF Millions)', fontsize=14)
		ax2b.set_ylabel('Reported FX Transactions (CHF Millions)', fontsize=14)
		ax2b.tick_params(axis='both', which='major', labelsize=12)
		for spine in ['top','right']:
			ax2b.spines[spine].set_visible(False)
		ax2b.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax2b.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax2b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=12)
		ax2b.grid(False)
		plt.tight_layout(); plt.show()

		# 3) Monthly bar chart (past 24 months)
		if self.monthly_interventions is not None:
			monthly_df = self.monthly_interventions.copy()
			monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
			cutoff = pd.Timestamp.now() - pd.DateOffset(months=24)
			recent = monthly_df[monthly_df['Date'] >= cutoff].copy()
			recent = recent.sort_values('Date')
			recent['MonthLabel'] = recent['Date'].dt.strftime('%Y-%m')
			fig3, ax3 = plt.subplots(figsize=(14, 6))
			xm = range(len(recent))
			ax3.bar(xm, recent['Predicted_Intervention'], color=colors['predicted'], alpha=0.8, edgecolor='black', linewidth=0.5)
			ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
			tick_pos = [i for i in xm if i % 3 == 0]
			ax3.set_xticks(tick_pos)
			ax3.set_xticklabels([recent['MonthLabel'].iloc[i] for i in tick_pos], rotation=45, ha='right', fontsize=11)
			ax3.set_ylabel('Monthly Flow (CHF Millions)', fontsize=14)
			ax3.tick_params(axis='y', which='major', labelsize=12)
			for spine in ['top','right']:
				ax3.spines[spine].set_visible(False)
		ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
		ax3.grid(False)
		plt.tight_layout(); plt.show()

	def create_monthly_imf_vs_fx_investments_plot(self) -> None:
		"""Plot monthly IMF reserves level against monthly balance sheet FX investments level (from 2020 onwards)."""
		if self.imf_reserves is None or self.monthly_fx_investments is None:
			return
		imf = self.imf_reserves.copy()
		fx = self.monthly_fx_investments.copy()
		# Standardize columns
		imf['Date'] = pd.to_datetime(imf['Date'])
		fx['Date'] = pd.to_datetime(fx['Date'])
		# Expect column name 'Foreign Currency Investments (CHF Millions)' for fx investments
		fx_col = [c for c in fx.columns if 'Foreign Currency Investments' in c]
		if not fx_col:
			return
		fx_level_col = fx_col[0]
		fx.rename(columns={fx_level_col: 'FX_Investments_Level'}, inplace=True)
		# Merge monthly
		merged = imf.merge(fx[['Date','FX_Investments_Level']], on='Date', how='left')
		merged = merged[merged['Date'] >= pd.to_datetime('2020-01-01')].copy()
		merged.sort_values('Date', inplace=True)
		# Build x-axis labels (show every 3rd month to reduce clutter)
		merged['MonthLabel'] = merged['Date'].dt.strftime('%Y-%m')
		x = range(len(merged))
		colors = {'imf':'#9467BD', 'fx':'#2CA02C'}
		fig, ax = plt.subplots(figsize=(12,8))
		ax.plot(x, merged['IMF_Reserves'], label='IMF Reserves Level', color=colors['imf'], linewidth=2.2)
		ax.plot(x, merged['FX_Investments_Level'], label='FX Investments Level', color=colors['fx'], linewidth=2.2, linestyle='--')
		# Tick formatting
		tick_positions = [i for i in x if i % 3 == 0]
		ax.set_xticks(tick_positions)
		ax.set_xticklabels([merged['MonthLabel'].iloc[i] for i in tick_positions], rotation=45, ha='right')
		ax.set_title('Monthly IMF Reserves vs FX Investments Level', fontweight='bold')
		ax.set_ylabel('CHF Millions')
		ax.grid(False)
		ax.legend()
		plt.tight_layout(); plt.show()

def create_asset_side_second_analyzer() -> AssetSideSecondAnalyzer:
	return AssetSideSecondAnalyzer()


def run_full_asset_side_second_analysis(processed_data: Dict[str, pd.DataFrame], monthly: bool = False) -> pd.DataFrame:
	analyzer = create_asset_side_second_analyzer()
	analyzer.load_prediction_data(processed_data)
	analyzer.load_market_data()
	analyzer.compute_weights()
	analyzer.compute_monthly_returns()
	analyzer.estimate_valuation_effects()
	analyzer.compute_interventions()
	if monthly:
		# Return monthly interventions without merging reported FX transactions (no aggregation/plots)
		return analyzer.monthly_interventions.copy()
	analyzer.aggregate_quarterly()
	comparison_df = analyzer.compare_with_reported_transactions()
	analyzer.create_comparison_visualization(comparison_df)

