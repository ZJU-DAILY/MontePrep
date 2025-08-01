from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path
from src.mcts.get_prompt import *
import os
import pandas as pd
import ast
import json

class RewardModel:
    def __init__(self, **kwargs):
        pass
    
    def get_reward(self, end_node) -> float:
        pass
    
    def execute_transformation(self, folder_path, transformation):
        """
        Execute the transformation and return the result.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class llmRewardModel(RewardModel):
    def __init__(self, llm_kwargs: Dict[str, Any]):
        self.llm_kwargs = llm_kwargs
        
    def get_reward(self, end_node, llm_client, method="columns_match") -> float:
        """
        Calculate reward based on the specified method.
        :param end_node: The end node containing transformation and schema information.
        :param llm_client: The LLM client for generating responses.
        :param method: The method for reward calculation. Options: "default", "llm_only", "columns_match".
        :return: Reward value as a float.
        """
        transformation = end_node.final_transformation
        table_schema = end_node.table_schema_dict
        result_table, _, columns_match, column_similarity = llmRewardModel.execute_transformation(end_node.table_path, transformation)

        if method == "llm_only":
            prompt = get_reward_prompt(
                table_schema_dict=table_schema,
                transformation=transformation,
                resulting_table="",
                column_match_str=""
            )
            responses, has_error = llm_client.generate_response(prompt, n=1)
            if not responses:
                return 0.0
            else:
                response = responses[0]
                try:
                    response = response.get("content", "")
                    start = response.find('```json')
                    end = response.find('```', start + 1)
                    if start != -1 and end != -1:
                        json_str = response[start + 7:end].strip()
                    else:
                        json_str = response.strip('` \n')
                    data = json.loads(json_str)
                    reward_str = str(data.get("reward", "0"))
                    if reward_str == "1":
                        return 1.0
                    elif reward_str == "0.5":
                        return 0.5
                    else:
                        return 0.0
                except Exception:
                    return 0.0

        elif method == "columns_match":
            return column_similarity
            
        elif method == "default":
            column_match_str = "The columns of the result table and the target table match." if columns_match else "The columns of the result table and the target table do not match."
            prompt = get_reward_prompt(
                table_schema_dict=table_schema,
                transformation=transformation,
                resulting_table=result_table,
                column_match_str=column_match_str
            )
            responses, has_error = llm_client.generate_response(prompt, n=1)
            if not responses:
                return 0.0
            else:
                response = responses[0]
                try:
                    response = response.get("content", "")
                    start = response.find('```json')
                    end = response.find('```', start + 1)
                    if start != -1 and end != -1:
                        json_str = response[start + 7:end].strip()
                    else:
                        json_str = response.strip('` \n')
                    data = json.loads(json_str)
                    reward_str = str(data.get("reward", "0"))
                    if reward_str == "1":
                        return 1.0
                    elif reward_str == "0.5":
                        return 0.5
                    else:
                        return 0.0
                except Exception:
                    return 0.0
        else:
            raise ValueError(f"Unknown reward calculation method: {method}")
    
    @staticmethod
    def execute_transformation(folder_path, transformation):
        table_dict = {}
        target_columns = None
        if 'length' in folder_path.name:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.csv') and not file_name.startswith('training'):
                    key = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    if key == "target":
                        df = pd.read_csv(file_path, nrows=5).iloc[:, 1:]
                        table_dict[key] = df
                        target_columns = list(df.columns)
                    else:
                        table_dict[key] = pd.read_csv(file_path).iloc[:, 1:]
        elif 'group' in folder_path.name:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.csv'):
                    key = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    if not key.startswith("target"):
                        table_dict['test_0'] = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path, nrows=5)
                        table_dict[key] = df
                        target_columns = list(df.columns)
        
        local_vars = {}
        local_vars.update(table_dict)
        final_df = None
        error_info = ""
        exec_env = {'pd': pd, **local_vars}
        try:
            for code in transformation:
                if not code:
                    continue
                exec(code, exec_env)
            last_var = llmRewardModel.extract_last_variable('\n'.join(transformation))
            final_df = exec_env.get(last_var)
        except Exception as e:
            error_info = str(e)
        result_table = ""
        column_similarity = 0.0
        columns_match = False
        if final_df is not None:
            caption = f"**Table Caption:** result"
            columns = "**Columns:**\n" + "\n".join([f"- {col}" for col in final_df.columns])
            result_table = f"{caption}\n{columns}\n\n"
            if target_columns is not None:
                columns_match = set(final_df.columns) == set(target_columns)
                column_similarity = len(set(final_df.columns) & set(target_columns)) / len(set(target_columns))
        return result_table, error_info, columns_match, column_similarity

    @staticmethod
    def extract_last_variable(code_str):
        tree = ast.parse(code_str)
        last_var = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in reversed(node.targets):
                    if isinstance(target, ast.Name):
                        last_var = target.id
                        break
        return last_var
