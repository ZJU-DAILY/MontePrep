import ast
import os
import pandas as pd
import json
from pandas.testing import assert_frame_equal
import numpy as np
from typing import Dict, Tuple
import random 

global_accuracy = {
    "total_samples": 0,
    "correct_total": 0
}
global_column_similarity = {
    "total_similarity": 0.0,
    "total_samples": 0
}

def read_csv_files(folder_path, folder_name):
    table_dict = {}
    if folder_name == "auto_pipeline":
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.csv') and not file_name.startswith('training'):
                key = os.path.splitext(file_name)[0]
                file_path = os.path.join(folder_path, file_name)
                table_dict[key] = pd.read_csv(file_path).iloc[:, 1:]
    elif folder_name == "buildings":
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.csv') and not file_name.startswith('target'):
                file_path = os.path.join(folder_path, file_name)
                table_dict['test_0'] = pd.read_csv(file_path)
    return table_dict
            
def calculate_similarity(result, target, rtol=1e-5, atol=1e-8):
        if result.empty or target.empty:
            return 0.0
        common_cols = list(set(result.columns) & set(target.columns))
        if not common_cols:
            return 0.0
        col_ratio = len(common_cols) / len(target.columns)
        
        result_common = result[common_cols].reset_index(drop=True)
        target_common = target[common_cols].reset_index(drop=True)
        sort_col = common_cols[0]
        result_sorted = result_common.sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
        target_sorted = target_common.sort_values(by=sort_col, key=lambda x: x.astype(str)).reset_index(drop=True)
        min_len = min(len(result_sorted), len(target_sorted))
        result_sorted = result_sorted.iloc[:min_len].reset_index(drop=True)
        result_sorted = target_sorted.iloc[:min_len].reset_index(drop=True)
        col_ratio = col_ratio * min_len / max(len(result_sorted), len(target_sorted))
        try:
            assert_frame_equal(result_sorted, target_sorted, 
                             check_exact=False, 
                             rtol=rtol, atol=atol,
                             check_dtype=False)
            return col_ratio
        except AssertionError:
            numeric_cols = result_sorted.select_dtypes(include=[np.number]).columns.tolist()
            str_cols = result_sorted.select_dtypes(include=['object']).columns.tolist()
            numeric_mask = np.isclose(
                result_sorted[numeric_cols], 
                target_sorted[numeric_cols], 
                rtol=rtol, 
                atol=atol, 
                equal_nan=True
            )
            def compare_str_cols(a, b):
                return (a.isna() & b.isna()) | (a == b)
            
            str_mask = compare_str_cols(
                result_sorted[str_cols],
                target_sorted[str_cols]
            ).to_numpy()
            combined_mask = np.concatenate([numeric_mask, str_mask], axis=1)
            similarity = np.mean(combined_mask)
            return similarity

def calculate_column_similarity(result, target):
    """Calculate the proportion of correctly generated columns."""
    if result.empty or target.empty:
        return 0.0
    common_cols = list(set(result.columns) & set(target.columns))
    return len(common_cols) / len(target.columns)
        
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

def get_output_var(code_lines):
    if not code_lines:
        return None
    last_line = code_lines[-1]
    if '=' in last_line:
        var_part = last_line.split('=', 1)[0].strip()
        return var_part
    else:
        return None
def process_json_files(
    json_folder: str, 
    folder_name: str,
    data_folder: str, 
    output_base: str, 
    length_type: int,
    start_num: int,
    end_num: int
) -> Tuple[Dict, Dict]:
    global global_accuracy, global_column_similarity 
    results = {}
    total_samples = 0
    correct_total = 0 
    total_column_similarity = 0
    
    for num in range(start_num, end_num):
        
        if folder_name == "auto_pipeline":
            target_file = os.path.join(data_folder, f"length{length_type}_{num}", "target.csv")
            folder_path = os.path.join(data_folder, f"length{length_type}_{num}")
            output_path = os.path.join(output_base, f"length{length_type}","tables")
            json_file = f"length{length_type}_{num}.json"
            os.makedirs(output_path, exist_ok=True)
        elif folder_name == "buildings":
            target_file = os.path.join(data_folder, f"group{length_type}_{num}", f"target{length_type}_{num}.csv")
            folder_path = os.path.join(data_folder, f"group{length_type}_{num}")
            output_path = os.path.join(output_base, f"group{length_type}","tables")
            json_file = f"group{length_type}_{num}.json"
            os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(target_file):
            continue
        json_path = os.path.join(json_folder, json_file)
        
        table_dict = read_csv_files(folder_path, folder_name)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            total_samples += 1
            continue
        paths = data 
        
        path = paths[0]
        similarity = 0.0
        exec_env = {'pd': pd, **table_dict}
        try:
            last_var = None
            code_str = None
            for code_line in path:
                code_str = code_line
                exec(code_str, exec_env)
                last_var = extract_last_variable(code_str)

            result = exec_env.get(last_var, pd.DataFrame())
            if folder_name == "auto_pipeline":
                target = pd.read_csv(target_file).iloc[:, 1:]
            else:
                target = pd.read_csv(target_file)
            similarity = calculate_similarity(result, target)
            column_similarity = calculate_column_similarity(result, target)
        except Exception as e:
            column_similarity = 0
        total_column_similarity += column_similarity
        global_column_similarity["total_similarity"] += column_similarity
        global_column_similarity["total_samples"] += 1  
        
        if similarity == 1.0:
            correct_total += 1
        total_samples += 1


    global_accuracy["total_samples"] += total_samples
    global_accuracy["correct_total"] += correct_total


    total_acc = correct_total / total_samples if total_samples else 0.0
    average_column_similarity = total_column_similarity / total_samples if total_samples else 0.0

    return {
        "accuracy": {
            "total_accuracy": total_acc,
            "total_samples": total_samples,
            "correct_total": correct_total,
            "average_column_similarity": average_column_similarity
        }
    }

def main(json_folder, data_folder, output_base, length_types, start_num, end_num):
    global global_accuracy  
    folder_name = os.path.basename(data_folder)
    if folder_name not in ["auto_pipeline", "buildings"]:
            raise ValueError(f"Unsupported folder name: {folder_name}")
        
    for length_type in length_types:
        if folder_name == "auto_pipeline":
            json_dir = os.path.join(json_folder, f"length{length_type}")
            output_dir = os.path.join(output_base, f"length{length_type}")
        elif folder_name == "buildings":
            json_dir = os.path.join(json_folder, f"group{length_type}")
            output_dir = os.path.join(output_base, f"group{length_type}")
        os.makedirs(output_dir, exist_ok=True)
        

        result_data = process_json_files(
            json_folder=json_dir,
            folder_name=folder_name,
            data_folder=data_folder,
            output_base=output_base,
            length_type=length_type,
            start_num=start_num,
            end_num=end_num
        )
        

        detail_path = os.path.join(output_dir, 'detail.json')
        accuracy_path = os.path.join(output_dir, 'accuracy.json')
        
        # with open(detail_path, 'w') as f:
        #     json.dump(result_data["results"], f, indent=4)
            
        with open(accuracy_path, 'w') as f:
            json.dump(result_data["accuracy"], f, indent=4)
            print(f"Length Type {length_type} Accuracy: {result_data['accuracy']['total_accuracy']:.2f}")
            print(f"Length Type {length_type} Column Similarity: {result_data['accuracy']['average_column_similarity']:.2f}")  # 新增：打印Column Similarity

    global_total_accuracy = (
        global_accuracy["correct_total"] / global_accuracy["total_samples"]
        if global_accuracy["total_samples"] else 0.0
    )
    global_total_column_similarity = (
        global_column_similarity["total_similarity"] / global_column_similarity["total_samples"]
        if global_column_similarity["total_samples"] else 0.0
    )
    with open(os.path.join(output_base, 'global_accuracy.json'), 'w') as f:
        json.dump(global_accuracy, f, indent=4)
    with open(os.path.join(output_base, 'global_column_similarity.json'), 'w') as f:
        json.dump(global_column_similarity, f, indent=4)
    print(f"Global Total Accuracy: {global_total_accuracy:.4f}")
    print(f"Global Total Column Similarity: {global_total_column_similarity:.4f}")  # 新增：打印全局列相似度

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_folder', type=str, default='./result/auto_pipeline')
    parser.add_argument('--data_folder', type=str, default='./data/auto_pipeline')
    parser.add_argument('--output_base', type=str, default='./predict/auto_pipeline')
    parser.add_argument('--length_types', type=int, nargs='+', default=[6])
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=100)
    args = parser.parse_args()
    main(
        json_folder=args.json_folder,
        data_folder=args.data_folder,
        output_base=args.output_base,
        length_types=args.length_types,
        start_num=args.start_num,
        end_num=args.end_num
    )