from src.mcts.mcts import MCTSSolver
from src.mcts.reward import llmRewardModel
from src.mcts.data import DataProcessor
import os
import json
import logging
from src.llm import LLMClient
import time
import argparse
import yaml


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCTS ETL Pipeline")
    parser.add_argument("--base_path", type=str, default="./data/auto_pipeline", help="Base path of data folder")
    parser.add_argument("--result_dir", type=str, default="./result", help="Result save path")
    parser.add_argument("--length_type", nargs='+', type=int, default=[5], help="Length types to process, support multiple values separated by space")
    parser.add_argument("--start_num", type=int, default=2, help="Start folder number")
    parser.add_argument("--end_num", type=int, default=3, help="End folder number")
    parser.add_argument("--log_path", type=str, default="logs/mcts_error_log.txt", help="Log file path")
    return parser.parse_args()

def initialize_logging(log_path):
    """Initialize logging configuration"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("mcts_etl")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def check_data_path(data_path, data_type, length, num):
    """Check if the data path exists"""
    if data_type == "auto_pipeline":
        return os.path.exists(os.path.join(data_path, f"length{length}_{num}"))
    elif data_type == "buildings":
        return os.path.exists(os.path.join(data_path, f"group{length}_{num}"))
    return False

def handle_exception(logger, length, num, attempt, exception):
    """Unified exception handling"""
    logger.error(f"MCTS solve failed: length={length}, num={num}, attempt={attempt+1} - {str(exception)}", exc_info=True)

def main():
    args = parse_arguments()
    logger = initialize_logging(args.log_path)

    # Read llm_kwargs config
    config_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    llm_kwargs = config.get("model_kwargs", {})
    llm_client = LLMClient(model_name=llm_kwargs.get("model_name", "qwen2.5-coder-32b-instruct")) 
    base_path = args.base_path
    result_dir = args.result_dir
    length_type = args.length_type
    length_value = list(range(args.start_num, args.end_num))

    # Configure MCTSSolver parameters
    max_rollout_steps = 10  
    max_depth = 5           
    exploration_constant = 1.0
    reward_model = llmRewardModel(llm_kwargs)  

    solver = MCTSSolver(
        max_rollout_steps=max_rollout_steps,
        max_depth=max_depth,
        exploration_constant=exploration_constant,
        llm_kwargs=llm_kwargs,
        llm_client=llm_client,
        reward_model=reward_model,
        logger=logger
    )
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, base_path)

    data_type = os.path.basename(data_path)
    
    
    for length in length_type:
        time_records = {}
        if data_type == "auto_pipeline":
            result_path = os.path.join(result_dir, f"length{length}")
        elif data_type == 'buildings':
            result_path = os.path.join(result_dir, f"group{length}")
        os.makedirs(result_path, exist_ok=True)
        for num in length_value:
            start_time = time.time()
            if not check_data_path(data_path, data_type, length, num):
                continue
            attempt_success = False
            for attempt in range(3):
                try:
                    if data_type == "auto_pipeline":
                        logger.info(f"Start solving with MCTS for length={length}, num={num}, attempt={attempt+1}...")
                    elif data_type == "buildings":
                        logger.info(f"Start solving with MCTS for group={length}, num={num}, attempt={attempt+1}...")
                    result_code = solver.solve(
                        bath_path=data_path,
                        data_type=data_type,
                        length_type=length,
                        length_value=num
                    )
                    token_usage = llm_client.token_usage
                    attempt_success = True
                    break
                except Exception as e:
                    handle_exception(logger, length, num, attempt, e)
            if not attempt_success:
                continue
            elapsed_time = time.time() - start_time
            if data_type == "auto_pipeline":
                task_name = f"length{length}_{num}"
            elif data_type == "buildings":
                task_name = f"group{length}_{num}"
            time_records[task_name] = {
                "elapsed_time": elapsed_time,
                "token_usage": token_usage
            }
            llm_client.reset_token_usage()
            try:
                json_file = os.path.join(result_path, f"{task_name}.json")
                with open(json_file, 'w') as f:
                    json.dump(result_code, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved results to {json_file}")
            except Exception as e:
                logger.error(f"Failed to save JSON: {task_name} - {str(e)}")

        metrics_json_path = os.path.join(result_dir, f"metrics_{data_type}_{length}.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(time_records, f, indent=4)
        logger.info(f"Metrics records saved to {metrics_json_path}")

if __name__ == "__main__":
    main()