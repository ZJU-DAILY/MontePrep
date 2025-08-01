from typing import List, Optional, Dict, Any
from src.mcts.types import MCTSNodeType, NODE_TYPE_TO_VALID_ACTIONS, MCTSAction
from src.llm import LLMClient
import copy

def get_valid_action_space_for_node(node: "MCTSNode") -> List["MCTSAction"]:
    from src.mcts.action import (
        SchemaMatchAction, IdentifyColumnFunctionsAction, TransformationAction,
        TransformationRevisionAction, EndAction
    )
    if node.node_type.value == MCTSNodeType.ROOT.value:
        action_space_classes = [SchemaMatchAction, IdentifyColumnFunctionsAction, TransformationAction]
    elif node.node_type.value == MCTSNodeType.SCHEMA_MATCH.value:
        action_space_classes = [IdentifyColumnFunctionsAction, TransformationAction]
    elif node.node_type.value == MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS.value:
        action_space_classes = [SchemaMatchAction, TransformationAction]
    elif node.node_type.value == MCTSNodeType.TRANSFORMATION.value:
        if hasattr(node, "columns_match") and node.columns_match is True:
            action_space_classes = [EndAction]
        else:
            action_space_classes = [EndAction, TransformationRevisionAction]
    elif node.node_type.value == MCTSNodeType.REVISED_TRANSFORMATION.value:
        action_space_classes = [EndAction]
    else:
        action_space_classes = []
    history_actions_classes = [path_node.parent_action.__class__ for path_node in node.path_nodes if path_node.parent_action is not None]
    valid_action_space = [action_class() for action_class in action_space_classes if action_class not in history_actions_classes]
    return valid_action_space

class MCTSNode:
    def __init__(self,
                 node_type: "MCTSNodeType",
                 parent_node: Optional["MCTSNode"] = None,
                 parent_action: Optional["MCTSAction"] = None,
                 depth: int = 0,
                 table_schema_dict: Optional[Dict[str, Any]] = None,
                 table_path: Optional[str] = None,
                 path_nodes: List["MCTSNode"] = [],
                 schema_match: Optional[Dict[str, Any]] = None,
                 column_functions: Optional[str] = None,
                 transformation: Optional[str] = None,
                 revised_transformation: Optional[str] = None,
                 final_transformation: Optional[str] = None,
                 is_valid_transformation: Optional[bool] = None,
                 llm_client: Optional[LLMClient] = None,
                 llm_kwargs: Optional[Dict[str, Any]] = None,
                 columns_match: Optional[bool] = None
                 ):
        self.node_type = node_type
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.depth = depth
        self.table_schema_dict = table_schema_dict
        self.table_path = table_path
        self.children : List[MCTSNode] = []
        self.path_nodes = path_nodes
        
        self.schema_match = schema_match
        self.column_functions = column_functions
        self.transformation = transformation
        self.revised_transformation = revised_transformation
        self.final_transformation = final_transformation
        self.is_valid_transformation = is_valid_transformation
        self.llm_kwargs = llm_kwargs
        
        self.llm_client = llm_client

        self.columns_match = columns_match

        self.Q = 0
        self.N = 0
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "llm_client":
                setattr(result, k, v) 
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def create_children(self):
        if self.children:
            return
        
        valid_action_space = get_valid_action_space_for_node(self)
        for action in valid_action_space:
            self.children.extend(action.create_children_nodes(self, self.llm_kwargs))
            
    def is_terminal(self):
        return getattr(self.node_type, "name", None) == "END"

