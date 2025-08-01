from src.mcts.node import *
from src.mcts.action import *
from src.mcts.reward import RewardModel
import math
import random
from pathlib import Path
from typing import Dict, Any, List
from src.mcts.data import DataProcessor
import pickle
import logging

class MCTSSolver:
    def __init__(self,
                 max_rollout_steps: int,
                 max_depth: int,
                 exploration_constant: float,
                 llm_kwargs: Dict[str, Any],
                 llm_client: LLMClient,
                 reward_model: RewardModel,
                 logger=None):  
        self.llm_client = llm_client
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.max_rollout_steps = max_rollout_steps
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.best_paths = []  
        self.logger = logger or logging.getLogger()  
    
    def log_info(self, message: str):
        self.logger.info(message)

    def should_terminate(self) -> bool:
        if len(self.best_paths) >= 2:
            self.log_info("Found 2 path with reward 1.0, terminating early.")
            return True
        return False

    def generate_folder_path(self, bath_path: str, data_type: str, length: int, num: int) -> Path:
        if data_type == "auto_pipeline":
            return Path(bath_path) / f"length{length}_{num}"
        elif data_type == "buildings":
            return Path(bath_path) / f"group{length}_{num}"
        return None

    def select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.is_terminal():
            if not all(child.N > 0 for child in current.children):
                return next(child for child in current.children if child.N == 0)
            
            current = max(current.children, key=lambda child: (child.Q / child.N) + self.exploration_constant * math.sqrt(math.log(current.N) / child.N))
        return current
    
    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        assert node.children == [], f"Children nodes of node {node.node_type} before expansion is not empty"
        valid_action_space = get_valid_action_space_for_node(node)
        for action in valid_action_space:
            action_nodes = action.create_children_nodes(node, self.llm_kwargs, logger=self.logger)  # 传递logger
            node.children.extend(action_nodes)
        random.shuffle(node.children)

    def simulate(self, node: MCTSNode) -> MCTSNode:
        assert node.children == [], f"Node before simulation have non-empty children"
        current = node
        
        expanded_nodes = []
        
        while not current.is_terminal():
            self.expand(current)
            expanded_nodes.append(current)
            current = random.choice(current.children)
            
        return current, expanded_nodes

    def backpropagate(self, node: MCTSNode):
        current = node
        reward = self.reward_model.get_reward(current, self.llm_client)
        if reward == 1.0:
            # pass
            self.best_paths.append(node.path_nodes)
        while current is not None:
            current.N += 1
            current.Q += reward
            current = current.parent_node
    
    def find_all_end_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        if node.node_type.value == MCTSNodeType.END.value:
            return [node]
        else:
            end_nodes = []
            for child in node.children:
                end_nodes.extend(self.find_all_end_nodes(child))
            return end_nodes
    
    def find_all_valid_reasoning_paths(self, node: MCTSNode) -> List[List[MCTSNode]]:
        end_nodes = self.find_all_end_nodes(node)
        node_scores = []
        for end_node in end_nodes:
            avg_score = end_node.Q / end_node.N if end_node.N > 0 else 0
            node_scores.append((avg_score, end_node.path_nodes))
        node_scores.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in node_scores]
    
    def solve(self, bath_path, data_type, length_type, length_value=None):
        if isinstance(length_value, List) or isinstance(length_type, List):
            self.logger.error("length_value should be a single integer, not a list.")
        length = length_type
        num = length_value

        self.best_paths = []
        meta_path = None
        folder_path = self.generate_folder_path(bath_path, data_type, length, num)
        if data_type == 'buildings':
            meta_path = folder_path / "meta.json"
        
        data_processor = DataProcessor(folder_path, data_type, meta_path)
        table_schema_dict = data_processor.process_tables()
        table_schema_dict_str = f"Source Tables:\n{table_schema_dict['source_tables']}\n Source Data Description:\n{table_schema_dict['source_data_description']}\n\nTarget Table:\n{table_schema_dict['target_table']}\nTarget Data Description:\n{table_schema_dict['target_data_description']}"
        root_node = MCTSNode(MCTSNodeType.ROOT,
                            parent_node=None,
                            parent_action=None,
                            depth=0,
                            table_schema_dict=table_schema_dict_str,
                            table_path=folder_path,
                            llm_client=self.llm_client,
                            llm_kwargs=self.llm_kwargs)
        root_node.path_nodes = [root_node]
        
        for _ in range(self.max_rollout_steps):
            if self.should_terminate():
                break
            self.log_info(f"Rollout step: {_ + 1}/{self.max_rollout_steps}")
            leaf_node = self.select(root_node)
            if leaf_node.is_terminal():
                self.backpropagate(leaf_node)
                continue
            self.expand(leaf_node)
            leaf_node = random.choice(leaf_node.children)
            end_node, simulated_expanded_nodes = self.simulate(leaf_node)
            
            self.backpropagate(end_node)
            
            # for n in simulated_expanded_nodes:
            #     n.children = []
                
            if self.should_terminate():
                break

        if len(self.best_paths) >= 2:
            all_valid_reasoning_paths = self.best_paths[:2]
        else:
            needed = 2 - len(self.best_paths)
            found_paths = self.find_all_valid_reasoning_paths(root_node)
            unique_paths = []
            for path in found_paths:
                if path not in self.best_paths:
                    unique_paths.append(path)
                if len(unique_paths) >= needed:
                    break
            all_valid_reasoning_paths = self.best_paths + unique_paths
        final_transformations = []
        for path in all_valid_reasoning_paths:
            if path and hasattr(path[-1], "final_transformation"):
                final_transformations.append(path[-1].final_transformation)
        return final_transformations

