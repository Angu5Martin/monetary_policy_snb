
import io
import pandas as pd
import requests
from typing import Optional, Dict
import warnings

from ..config import (
    BASE_URL, CUBE_BALANCE_SHEET_ITEMS, CUBE_FX_TRANSACTIONS, CUBE_FX_INVESTMENTS,
    CUBE_FX_RATES, CUBE_CHF_INDEX, CUBE_DIMENSION_SELECTIONS, REQUEST_TIMEOUT,
    CUBE_IMF_RESERVES
)

warnings.filterwarnings('ignore')


class SNBDataCollector:
    
    def __init__(self):
        self.base_url = BASE_URL
        self.timeout = REQUEST_TIMEOUT
    
    def fetch_snb_cube(self, cube_id: str, from_date: Optional[str] = None, 
                      to_date: Optional[str] = None, lang: str = "en") -> pd.DataFrame:
 
        params = {}
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date
        
        # Add dimension selection for specific cubes
        if cube_id in CUBE_DIMENSION_SELECTIONS:
            params["dimSel"] = CUBE_DIMENSION_SELECTIONS[cube_id]

        url = self.base_url.format(cube=cube_id, lang=lang)
        
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch data from SNB API for cube {cube_id}: {e}")

        lines = r.text.splitlines()
        
        # Find the header line (contains "Date")
        header_idx = None
        for i, line in enumerate(lines):
            if '"Date"' in line and ";" in line:
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError(f"Could not find header row in SNB data for cube {cube_id}")
        
        csv_content = '\n'.join(lines[header_idx:])
        
        df = pd.read_csv(
            io.StringIO(csv_content),
            sep=";",
            quotechar='"',
            na_values=['', ' '],
        )
        
        # Clean column names
        df.columns = [c.strip().strip('"') for c in df.columns]
        
        # Convert Value column to numeric
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        return df
    
    def collect_balance_sheet_data(self, from_date: Optional[str] = None, 
                                 to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_BALANCE_SHEET_ITEMS, from_date, to_date)
    
    def collect_fx_transactions_data(self, from_date: Optional[str] = None, 
                                   to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_FX_TRANSACTIONS, from_date, to_date)
    
    def collect_fx_investments_data(self, from_date: Optional[str] = None, 
                                  to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_FX_INVESTMENTS, from_date, to_date)
    
    def collect_fx_rates_data(self, from_date: Optional[str] = None, 
                            to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_FX_RATES, from_date, to_date)
    
    def collect_chf_index_data(self, from_date: Optional[str] = None, 
                             to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_CHF_INDEX, from_date, to_date)

    def collect_imf_reserves_data(self, from_date: Optional[str] = None, 
                                to_date: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_snb_cube(CUBE_IMF_RESERVES, from_date, to_date)

    def collect_all_snb_data(self, from_date: Optional[str] = None, 
                           to_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
          
        data = {}
        
        data['balance_sheet'] = self.collect_balance_sheet_data(from_date, to_date)
        
        data['fx_transactions'] = self.collect_fx_transactions_data(from_date, to_date)
        
        data['fx_investments'] = self.collect_fx_investments_data(from_date, to_date)
        
        data['fx_rates'] = self.collect_fx_rates_data(from_date, to_date)
        
        data['chf_index'] = self.collect_chf_index_data(from_date, to_date)

        data['imf_reserves'] = self.collect_imf_reserves_data(from_date, to_date)

        return data


def collect_all_snb_data(from_date: Optional[str] = None, 
                        to_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    collector = SNBDataCollector()
    return collector.collect_all_snb_data(from_date, to_date)
