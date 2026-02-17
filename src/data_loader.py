# src/data_loader.py
import pandas as pd
import yaml
import logging
from pathlib import Path

# Set up basic logging for MLOps tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WWTPDataLoader:
    """
    Data Loader class to ingest raw SCADA data and format the time-series index,
    controlled entirely by the config.yaml file.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Reads and parses the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        with open(self.config_path, "r") as file:
            try:
                config = yaml.safe_load(file)
                logging.info(f"Configuration loaded successfully from {self.config_path}")
                return config
            except yaml.YAMLError as exc:
                logging.error(f"Error parsing YAML file: {exc}")
                raise

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads the raw CSV file, cleans up artifact columns, and sets 
        the datetime index based on parameters defined in the config file.
        """
        raw_path = Path(self.config['data']['raw_path'])
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_path}. Ensure it is downloaded.")

        logging.info(f"Loading raw dataset from {raw_path}...")
        df = pd.read_csv(raw_path)
        
        # Drop CSV artifact column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            logging.info("Dropped 'Unnamed: 0' artifact column.")
        
        # Time-Series Index Formatting (Handling split Year/Month/Day columns)
        date_cols = self.config['datetime']['columns']
        
        # Check if all date columns exist in the dataframe
        if all(col in df.columns for col in date_cols):
            logging.info(f"Combining {date_cols} into datetime index...")
            
            # Pandas to_datetime can combine Year, Month, Day columns automatically
            # We cast to int first to remove the float decimals (e.g., 2014.0 -> 2014)
            df['Date'] = pd.to_datetime(df[date_cols].astype(int))
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Drop the original redundant columns to keep the dataframe clean
            df.drop(columns=date_cols, inplace=True)
            logging.info("Time-series index set successfully.")
        else:
            logging.warning(f"Required date columns {date_cols} not found. Skipping index formatting.")
            
        return df