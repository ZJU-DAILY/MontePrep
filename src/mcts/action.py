from src.mcts.types import MCTSNodeType, MCTSAction, NODE_TYPE_TO_VALID_ACTIONS
from src.mcts.node import *
from src.llm import LLMClient
from typing import Dict, Any, List, Optional, Tuple 
from enum import Enum
from pathlib import Path
from collections import defaultdict
from src.mcts.get_prompt import *
from src.mcts.reward import *
import copy
import json
import re
import random
import logging


class MCTSNodeType(Enum):
    ROOT = "root"
    SCHEMA_MATCH= "schema_match"
    IDENTIFY_COLUMN_FUNCTIONS = "identify_column_functions"
    TRANSFORMATION= "transformation"
    REVISED_TRANSFORMATION = "revised_transformation"
    END = "end"
    
    
class MCTSAction:
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        raise NotImplementedError()


class SchemaMatchAction(MCTSAction):
    """
    Select the schema context that is most relevant to the question.
    
    Valid previous nodes:
    - Root node
    - Identify column functions node
    """
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        table_schema = node.table_schema_dict
        previous_thoughts = ""
        for path_node in node.path_nodes:
            if isinstance(path_node.parent_action, IdentifyColumnFunctionsAction):
                previous_thoughts += f"Possible column functions: {path_node.column_functions}\n"
        hint = f"\n\nHere are my previous thoughts:\n{previous_thoughts}" if previous_thoughts else ""
        prompt = get_schema_match_prompt(table_schema_dict=table_schema, hint=hint)

        nodes = []
        while len(nodes) < llm_kwargs["n"]:
            new_max_gen_nums = llm_kwargs["n"] - len(nodes)
            responses, error = node.llm_client.generate_response(prompt, n=new_max_gen_nums)
            if error:
                if logger:
                    logger.warning(f"Error generating schema match response: {responses}")
                else:
                    print(f"Error generating schema match response: {responses}")
            for resp in responses:
                response = resp["content"]
                child_node = copy.deepcopy(node)
                child_node.node_type = MCTSNodeType.SCHEMA_MATCH
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.children = []
                child_node.path_nodes = node.path_nodes + [child_node]
                new_schema_match = self.schema_match(response)
                if new_schema_match:
                    child_node.schema_match = new_schema_match
                nodes.append(child_node)
        return nodes[:llm_kwargs["n"]]
    
    def schema_match(self, response: str):
        try:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                json_str = json_str.replace('{{', '{').replace('}}', '}')
                return json_str
            else:
                print("JSON content not found")
                return ""
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return ""


class IdentifyColumnFunctionsAction(MCTSAction):
    """
    Identify the column functions that are most relevant to the question.
    
    Valid previous nodes:
    - Root node
    - Schema match node
    """
    
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        table_schema = node.table_schema_dict
        
        previous_thoughts = ""
        for path_node in node.path_nodes:
            if isinstance(path_node.parent_action, SchemaMatchAction):
                previous_thoughts += f"Possible schema match info: {path_node.schema_match}\n"
        hint = f"\n\nHere are my previous thoughts:\n{previous_thoughts}" if previous_thoughts else ""
        prompt = get_identify_function_prompt(table_schema_dict=table_schema, hint=hint)
        
        responses, error = node.llm_client.generate_response(prompt, n=llm_kwargs["n"])
        if error:
            if logger:
                logger.warning(f"Error generating identify column functions response: {responses}")
            else:
                print(f"Error generating identify column functions response: {responses}")
        contents = list(set([resp["content"] for resp in responses]))
        nodes = []
        for response in contents:
            child_node = copy.deepcopy(node)
            child_node.node_type = MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS
            child_node.parent_node = node
            child_node.parent_action = self
            child_node.depth = node.depth + 1
            child_node.children = []
            child_node.path_nodes = node.path_nodes + [child_node]
            child_node.column_functions = response
            nodes.append(child_node)
        return nodes



class TransformationAction(MCTSAction):
    """
    Generate the SQL query.
    
    Valid previous nodes:
    - Root node
    - Schema match node
    - Identify column functions node
    """
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        table_schema = node.table_schema_dict
        previous_thoughts = ""
        for path_node in node.path_nodes:
            if isinstance(path_node.parent_action, IdentifyColumnFunctionsAction):
                previous_thoughts += f"Possible column functions: {path_node.column_functions}\n"
        hint = f"\n\nHere are my previous thoughts:\n{previous_thoughts}" if previous_thoughts else ""
        prompt = get_transformation_prompt(table_schema_dict=table_schema, hint=hint)
        
        nodes = []
        while len(nodes) < llm_kwargs["n"]:
            new_max_gen_nums = llm_kwargs["n"] - len(nodes)
            responses, error = node.llm_client.generate_response(prompt, n=new_max_gen_nums)
            if error:
                if logger:
                    logger.warning(f"Error generating transformation response: {responses}")
                else:
                    print(f"Error generating transformation response: {responses}")
            for resp in responses:
                response = resp["content"]
                child_node = copy.deepcopy(node)
                child_node.node_type = MCTSNodeType.TRANSFORMATION
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.children = []
                child_node.path_nodes = node.path_nodes + [child_node]
                tranformation = self.extract_tranformation_answer(response, node.llm_client)
                child_node.transformation = tranformation
                _, _, columns_match,_ = llmRewardModel.execute_transformation(child_node.table_path, tranformation)
                child_node.columns_match = columns_match
                nodes.append(child_node)
        return nodes
    
    def extract_tranformation_answer(self, response: str, llm_client: LLMClient) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, use llm_client.generate_response() with a new prompt to optimize it
            optimization_prompt = f"Please convert the following response into valid JSON format:\n\n{response}"
            optimized_response, error = llm_client.generate_response(optimization_prompt, n=1)
            if error or not optimized_response:
                print(f"Error optimizing response: {error}")
                data = {"code": []}
            else:
                try:
                    optimized_content = optimized_response[0]["content"]
                    data = json.loads(optimized_content)
                except json.JSONDecodeError:
                    data = {"code": []}
        return data.get("code", [])

class TransformationRevisionAction(MCTSAction):
    """
    Revise the SQL query with the given context.
    
    Valid previous nodes:
    - Schema match node
    - Identify column functions node
    - SQL generation node
    """
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        table_schema = node.table_schema_dict
        previous_thoughts = ""
        error_message = ""
        orginal_code = ""
        for path_node in node.path_nodes:
            if isinstance(path_node.parent_action, IdentifyColumnFunctionsAction):
                previous_thoughts += f"Identify column functions: {path_node.column_functions}\n"
            elif isinstance(path_node.parent_action, TransformationAction):
                execution_result, error_info, column_match,_ = llmRewardModel.execute_transformation(path_node.table_path, path_node.transformation)
                error_message = error_info if error_info else ""
                if not column_match:
                    error_message += "The Original code execution result does not match the target table schema.\n"
                orginal_code = path_node.transformation
        hint = f"\n\nHere are my previous thoughts:\n{previous_thoughts}" if previous_thoughts else ""
        prompt = get_transformation_revision_prompt(table_schema_dict=table_schema, hint=hint, original_code=orginal_code, error_message=error_message, exec_result=execution_result)
        nodes = []
        while len(nodes) < llm_kwargs["n"]:
            new_llm_kwargs = copy.deepcopy(llm_kwargs)
            new_llm_kwargs["n"] = llm_kwargs["n"] - len(nodes)
            responses, error = node.llm_client.generate_response(prompt, n=new_llm_kwargs["n"])
            if error:
                if logger:
                    logger.warning(f"Error generating transformation revision response: {responses}")
                else:
                    print(f"Error generating transformation revision response: {responses}")
            for resp in responses:
                response = resp["content"]
                child_node = copy.deepcopy(node)
                child_node.node_type = MCTSNodeType.REVISED_TRANSFORMATION
                child_node.parent_node = node
                child_node.parent_action = self
                child_node.depth = node.depth + 1
                child_node.children = []
                child_node.path_nodes = node.path_nodes + [child_node]
                revised_transformation = self.extract_tranformation_answer(response, node.llm_client)
                child_node.revised_transformation = revised_transformation
                nodes.append(child_node)
        return nodes

    def extract_tranformation_answer(self, response: str, llm_client: LLMClient) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, use llm_client.generate_response() with a new prompt to optimize it
            optimization_prompt = f"Please convert the following response into valid JSON format:\n\n{response}"
            optimized_response, error = llm_client.generate_response(optimization_prompt, n=1)
            if error or not optimized_response:
                print(f"Error optimizing response: {error}")
                data = {"code": []}
            else:
                try:
                    optimized_content = optimized_response[0]["content"]
                    data = json.loads(optimized_content)
                except json.JSONDecodeError:
                    data = {"code": []}
        return data.get("code", [])
    
    
class EndAction(MCTSAction):
    """
    End the search.
    """
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any], logger=None) -> List["MCTSNode"]:
        assert node.node_type == MCTSNodeType.TRANSFORMATION or node.node_type == MCTSNodeType.REVISED_TRANSFORMATION
        child_node = copy.deepcopy(node)
        child_node.node_type = MCTSNodeType.END
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1
        child_node.children = []
        child_node.path_nodes = node.path_nodes + [child_node]
        child_node.final_transformation = node.transformation if node.node_type == MCTSNodeType.TRANSFORMATION else node.revised_transformation
        return [child_node]



NODE_TYPE_TO_VALID_ACTIONS.update({
    MCTSNodeType.ROOT: [
        SchemaMatchAction,
        IdentifyColumnFunctionsAction,
        TransformationAction
    ],
    MCTSNodeType.SCHEMA_MATCH: [
        IdentifyColumnFunctionsAction,
        TransformationAction
    ],
    MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS: [
        SchemaMatchAction,
        TransformationAction
    ],
    MCTSNodeType.TRANSFORMATION: [
        EndAction,
        TransformationRevisionAction
    ],
    MCTSNodeType.REVISED_TRANSFORMATION: [
        EndAction
    ],
    MCTSNodeType.END: []
})