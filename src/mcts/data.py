import json
import os
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

class DataProcessor:
    def __init__(self, folder_path, data_type, meta_path = None):
        """
        """
        self.table_dict = {}
        self.schema_match = None
        self.meta_data = None
        self.data_type = data_type
        self._read_csv_files(folder_path)
        if meta_path:
            self._read_meta_file(meta_path)

    def _read_csv_files(self, folder_path):
        """
        """
        self.table_dict = {}
        if self.data_type == "auto_pipeline":
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.csv') and not file_name.startswith('training'):
                    key = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    if key == "target":
                        self.table_dict[key] = pd.read_csv(file_path, nrows=5).iloc[:, 1:]
                    else:
                        self.table_dict[key] = pd.read_csv(file_path).iloc[:, 1:]
        if self.data_type == "buildings":
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.csv'):
                    key = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    if not key.startswith("target"):
                        self.table_dict['test_0'] = pd.read_csv(file_path)
                    if key.startswith("target"):
                        self.table_dict['target'] = pd.read_csv(file_path)

    def read_schema_match(self,schema_match_path):
        with open(schema_match_path, 'r', encoding='utf-8') as file:
            data = json.load(file) 
        self.schema_match = json.dumps(data, ensure_ascii=False, indent=4)
        return self.schema_match
    
    def _read_meta_file(self, meta_path):
        """
        """
        with open(meta_path, 'r', encoding='utf-8') as file:
            self.meta_data = json.load(file)
            
    def process_tables(self):
        """
        """
        source_tables = []
        target_table = None

        for key, df in self.table_dict.items():
            caption = f"**Table Caption:** {key}"
            columns = "**Columns:**\n" + "\n".join([f"- {col}" for col in df.columns])
            
            rows = []
            for idx, (_, row) in enumerate(df.head(3).iterrows(), 1):
                row_str = " | ".join(map(str, row.values))
                rows.append(f"{idx}. | {row_str} |")
            rows_str = "**Rows:**\n" + "\n".join(rows)
            
            table_str = f"{caption}\n{columns}\n{rows_str}"
            
            if key == 'target':
                target_table = f"{caption}\n{columns}"
            else:
                source_tables.append(table_str)
        target_data_description = ""
        source_data_description = ""    
        if self.meta_data:
            target_table = f"**Table Caption:** {self.meta_data.get('Target Data Name','')}\n**Columns:**\n{self.meta_data.get('Target Data Schema', '')}"
            target_data_description = self.meta_data.get('Target Data Description', '')
            source_data_description = self.meta_data.get('Source Data Description', '')
        return {
            "source_tables": "\n".join(source_tables),
            "target_table": target_table,
            "target_data_description": target_data_description,
            "source_data_description": source_data_description
        }
