
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings

from ..config import FX_CURRENCY_MAPPING, CURRENCIES

warnings.filterwarnings('ignore')


class SNBDataPreprocessor:
    
    def __init__(self):
        self.fx_currency_mapping = FX_CURRENCY_MAPPING
        self.currencies = CURRENCIES
    
    def parse_dates(self, df: pd.DataFrame, end_of_period: bool = False) -> pd.DataFrame:

        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            if end_of_period:
                # For quarterly data, move to the last month of the quarter
                # Q1 (Jan/Feb/Mar) -> Mar, Q2 (Apr/May/Jun) -> Jun, etc.
                df['Date'] = df['Date'] + pd.offsets.QuarterEnd(0)
            
            # Format all dates as year-month strings (YYYY-MM)
            # This removes the day component entirely
            df['Date'] = df['Date'].dt.strftime('%Y-%m')
                
        return df
    
    def create_balance_sheet_dataframes(self, bsheet_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Parse dates first
        bsheet_data = self.parse_dates(bsheet_data)
        
        # Foreign Currency Investments (D)
        foreign_currency_investments_df = bsheet_data[bsheet_data['D0'] == 'D'][['Date', 'Value']].copy()
        foreign_currency_investments_df.columns = ['Date', 'Foreign Currency Investments (CHF Millions)']
        foreign_currency_investments_df = foreign_currency_investments_df.sort_values('Date').reset_index(drop=True)

        # Bank Sight Deposits (GB)
        bank_sight_deposits_df = bsheet_data[bsheet_data['D0'] == 'GB'][['Date', 'Value']].copy()
        bank_sight_deposits_df.columns = ['Date', 'Bank Sight Deposits (CHF Millions)']
        bank_sight_deposits_df = bank_sight_deposits_df.sort_values('Date').reset_index(drop=True)

        # Bank Notes in Circulation (N)
        bank_notes_in_circulation_df = bsheet_data[bsheet_data['D0'] == 'N'][['Date', 'Value']].copy()
        bank_notes_in_circulation_df.columns = ['Date', 'Bank Notes in Circulation (CHF Millions)']
        bank_notes_in_circulation_df = bank_notes_in_circulation_df.sort_values('Date').reset_index(drop=True)

        # Liabilities from CHF Repo Transactions (VRGSF)
        liabilities_chf_repo_df = bsheet_data[bsheet_data['D0'] == 'VRGSF'][['Date', 'Value']].copy()
        liabilities_chf_repo_df.columns = ['Date', 'Liabilities from CHF Repo Transactions (CHF Millions)']
        liabilities_chf_repo_df = liabilities_chf_repo_df.sort_values('Date').reset_index(drop=True)

        # SNB Debt Certificates (ES)
        snb_debt_certificates_df = bsheet_data[bsheet_data['D0'] == 'ES'][['Date', 'Value']].copy()
        snb_debt_certificates_df.columns = ['Date', 'SNB Debt Certificates (CHF Millions)']
        snb_debt_certificates_df = snb_debt_certificates_df.sort_values('Date').reset_index(drop=True)

        # Amount from the Confederation (VB)
        confederation_amount_df = bsheet_data[bsheet_data['D0'] == 'VB'][['Date', 'Value']].copy()
        confederation_amount_df.columns = ['Date', 'Amount from the Confederation (CHF Millions)']
        confederation_amount_df = confederation_amount_df.sort_values('Date').reset_index(drop=True)

        # Amount from Foreign Bank Sight Deposits (GBI)
        foreign_bank_sight_deposits_df = bsheet_data[bsheet_data['D0'] == 'GBI'][['Date', 'Value']].copy()
        foreign_bank_sight_deposits_df.columns = ['Date', 'Amount from Foreign Bank Sight Deposits (CHF Millions)']
        foreign_bank_sight_deposits_df = foreign_bank_sight_deposits_df.sort_values('Date').reset_index(drop=True)

        claims_chf_repo_df = bsheet_data[bsheet_data['D0'] == 'FRGSF'][['Date', 'Value']].copy()
        claims_chf_repo_df.columns = ['Date', 'Claims from CHF Repo Transactions (CHF Millions)']
        claims_chf_repo_df = claims_chf_repo_df.sort_values('Date').reset_index(drop=True)

        secured_loans_df = bsheet_data[bsheet_data['D0'] == 'GD'][['Date', 'Value']].copy()
        secured_loans_df.columns = ['Date', 'Secured Loans (CHF Millions)']
        secured_loans_df = secured_loans_df.sort_values('Date').reset_index(drop=True)

        fx_liabilities_df = bsheet_data[bsheet_data['D0'] == 'VF'][['Date', 'Value']].copy()
        fx_liabilities_df.columns = ['Date', 'FX Liabilities (CHF Millions)']
        fx_liabilities_df = fx_liabilities_df.sort_values('Date').reset_index(drop=True)

        gold_reserves_df = bsheet_data[bsheet_data['D0'] == 'GFG'][['Date', 'Value']].copy()
        gold_reserves_df.columns = ['Date', 'Gold Reserves (CHF Millions)']
        gold_reserves_df = gold_reserves_df.sort_values('Date').reset_index(drop=True)

        equity_provisions_df = bsheet_data[bsheet_data['D0'] == 'RE'][['Date', 'Value']].copy()
        equity_provisions_df.columns = ['Date', 'Equity Provisions (CHF Millions)']
        equity_provisions_df = equity_provisions_df.sort_values('Date').reset_index(drop=True)

        return foreign_currency_investments_df, bank_sight_deposits_df, bank_notes_in_circulation_df, \
               liabilities_chf_repo_df, snb_debt_certificates_df, foreign_bank_sight_deposits_df, \
               claims_chf_repo_df, secured_loans_df, fx_liabilities_df, gold_reserves_df, equity_provisions_df, confederation_amount_df

    def create_fx_investments_dataframe(self, fx_inv_data: pd.DataFrame) -> pd.DataFrame:

        # Parse dates for quarterly data (end of period)
        fx_inv_data = self.parse_dates(fx_inv_data, end_of_period=True)
        
        # Pivot the data to have currencies as columns
        fx_investments_pivot = fx_inv_data.pivot_table(
            index='Date', 
            columns='D0', 
            values='Value', 
            aggfunc='first'
        ).reset_index()
        
        # Rename columns using the currency mapping
        currency_columns = ['Date']
        for code, currency in self.fx_currency_mapping.items():
            if code in fx_investments_pivot.columns:
                fx_investments_pivot = fx_investments_pivot.rename(columns={code: currency})
                currency_columns.append(currency)
        
        # Select only the columns we need and reorder
        available_columns = ['Date'] + [col for col in self.currencies if col in fx_investments_pivot.columns]
        fx_investments_df = fx_investments_pivot[available_columns].copy()
        
        # Sort by date and reset index
        fx_investments_df = fx_investments_df.sort_values('Date').reset_index(drop=True)
        
        return fx_investments_df
    
    def create_fx_transactions_dataframe(self, fx_trx_data: pd.DataFrame) -> pd.DataFrame:

        # Parse dates for quarterly data (end of period)
        fx_trx_data = self.parse_dates(fx_trx_data, end_of_period=True)
        
        # Create transactions dataframe with just Date and Total
        fx_transactions_df = fx_trx_data[['Date', 'Value']].copy()
        fx_transactions_df.columns = ['Date', 'Total']
        fx_transactions_df = fx_transactions_df.sort_values('Date').reset_index(drop=True)
        
        return fx_transactions_df
    
    def create_fx_rates_dataframe(self, fx_rates_data: pd.DataFrame) -> pd.DataFrame:

        # Parse dates
        fx_rates_data = self.parse_dates(fx_rates_data)
        
        # Create mapping for FX rate columns
        fx_rate_mapping = {
            'EUR1': 'EUR',
            'GBP1': 'GBP', 
            'USD1': 'USD',
            'CAD1': 'CAD',
            'JPY100': 'JPY/100'
        }
        
        # Pivot the data
        fx_rates_pivot = fx_rates_data.pivot_table(
            index='Date',
            columns='D1', 
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        # Rename columns
        for old_name, new_name in fx_rate_mapping.items():
            if old_name in fx_rates_pivot.columns:
                fx_rates_pivot = fx_rates_pivot.rename(columns={old_name: new_name})
        
        # Sort by date and reset index
        fx_rates_df = fx_rates_pivot.sort_values('Date').reset_index(drop=True)
        
        return fx_rates_df
    
    def create_chf_index_dataframe(self, chf_index_data: pd.DataFrame) -> pd.DataFrame:

        # Parse dates
        chf_index_data = self.parse_dates(chf_index_data)
        
        # Create CHF index dataframe
        chf_index_df = chf_index_data[['Date', 'Value']].copy()
        chf_index_df.columns = ['Date', 'CHF_INDEX']
        chf_index_df = chf_index_df.sort_values('Date').reset_index(drop=True)
        
        return chf_index_df

    def create_imf_reserves_dataframe(self, imf_reserves_data: pd.DataFrame) -> pd.DataFrame:

        # Parse dates
        imf_reserves_data = self.parse_dates(imf_reserves_data)

        # Create IMF reserves dataframe
        imf_reserves_df = imf_reserves_data[['Date', 'Value']].copy()
        imf_reserves_df.columns = ['Date', 'IMF_Reserves']
        imf_reserves_df = imf_reserves_df.sort_values('Date').reset_index(drop=True)

        return imf_reserves_df

    def preprocess_all_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        
        processed_data = {}
        
        # Process balance sheet data
        fci_df, bsd_df, bn_df, lchf_df, sdbc_df, foreign_bank_sight_deposits_df, claims_chf_repo_df, secured_loans_df, fx_liabilities_df, gold_reserves_df, equity_provisions_df, confederation_amount_df = self.create_balance_sheet_dataframes(raw_data['balance_sheet'])
        processed_data['foreign_currency_investments'] = fci_df
        processed_data['bank_sight_deposits'] = bsd_df
        processed_data['bank_notes_in_circulation'] = bn_df
        processed_data['liabilities_chf_repo'] = lchf_df
        processed_data['snb_debt_certificates'] = sdbc_df
        processed_data['foreign_bank_sight_deposits'] = foreign_bank_sight_deposits_df
        processed_data['claims_chf_repo'] = claims_chf_repo_df
        processed_data['secured_loans'] = secured_loans_df
        processed_data['fx_liabilities'] = fx_liabilities_df
        processed_data['gold_reserves'] = gold_reserves_df
        processed_data['equity_provisions'] = equity_provisions_df
        processed_data['confederation_amount'] = confederation_amount_df

        # Process FX investments data
        processed_data['fx_investments'] = self.create_fx_investments_dataframe(raw_data['fx_investments'])
        
        # Process FX transactions data
        processed_data['fx_transactions'] = self.create_fx_transactions_dataframe(raw_data['fx_transactions'])
        
        # Process FX rates data
        processed_data['fx_rates'] = self.create_fx_rates_dataframe(raw_data['fx_rates'])
        
        # Process CHF index data
        processed_data['chf_index'] = self.create_chf_index_dataframe(raw_data['chf_index'])

        # Process IMF reserves data
        processed_data['imf_reserves'] = self.create_imf_reserves_dataframe(raw_data['imf_reserves'])
        
        return processed_data


def preprocess_all_snb_data(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:

    preprocessor = SNBDataPreprocessor()
    return preprocessor.preprocess_all_data(raw_data)
