import pandas as pd
import numpy as np

class DataContainer:
    def __init__(self, path):
        self.path: str = path
        self.data = None
        
        self.features = None
        self.label = None
        
        self.load_data()
        self.optimize_int()
        self.optimize_float()
        
        self.info
    
    def load_data(self) -> None:
        match self.path.split('.')[-1]:
            case 'csv':
                self.data = pd.read_csv(self.path)
            case 'xlsx':
                self.data = pd.read_excel(self.path)
            case 'json':
                self.data = pd.read_json(self.path)
            case 'parquet':
                self.data = pd.read_parquet(self.path)
            case 'feather':
                self.data = pd.read_feather(self.path)
            case 'html':
                self.data = pd.read_html(self.path)
            case 'pickle':
                self.data = pd.read_pickle(self.path)
            case 'sql':
                self.data = pd.read_sql(self.path)
            case 'hdf':
                self.data = pd.read_hdf(self.path)
            case 'stata':
                self.data = pd.read_stata(self.path)
            case 'sas':
                self.data = pd.read_sas(self.path)
            case 'spss':
                self.data = pd.read_spss(self.path)
            case 'fwf':
                self.data = pd.read_fwf(self.path)
            case 'clipboard':
                self.data = pd.read_clipboard(self.path)
            case _:
                raise ValueError('File type not supported. Refer to pandas documentation for supported file types.')
            
    def optimize_int(self) -> None:
        for col in self.data.select_dtypes(include=int).columns:
            self.data[col] = pd.to_numeric(self.data[col], downcast='integer')
            
    def optimize_float(self) -> None:
        for col in self.data.select_dtypes(include=float).columns:
            self.data[col] = pd.to_numeric(self.data[col], downcast='float')
    
    def encode_str(self, exclude:str) -> None:
        for col in self.data.select_dtypes(include=object).columns:
            if col != exclude:
                enum =  {val: i for i, val in enumerate(list(self.data[col].unique()))}
                self.data[col] = pd.to_numeric(self.data[col].map(enum), downcast='integer')
            else:
                pass
            
    def split_data(self, label: str, drop: list) -> None:
        if label in drop:
            raise ValueError('Label column cannot be dropped.')
        
        data = self.data.drop(drop, axis=1)
        self.features = self.data[data.columns.difference([label])]
        self.label = data[label]
    
    @property
    def head(self) -> None:
        print(self.data.head())
    
    @property
    def tail(self) -> None:
        print(self.data.tail())
    
    @property
    def info(self) -> None:
        print(self.data.info(memory_usage='deep'))
    
    @property
    def describe(self) -> None:
        print(self.data.describe())
    
    @property
    def shape(self) -> None:
        print(self.data.shape)
        
    @property
    def columns(self) -> None:
        print(self.data.columns)
    
    @property
    def len(self) -> None:
        print(len(self.data))