from enum import Enum
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.mcts.node import MCTSNode

class MCTSNodeType(Enum):
    ROOT = "root"
    SCHEMA_MATCH= "schema_match"
    IDENTIFY_COLUMN_FUNCTIONS = "identify_column_functions"
    TRANSFORMATION= "transformation"
    REVISED_TRANSFORMATION = "revised_transformation"
    END = "end"

class MCTSAction:
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        raise NotImplementedError()


NODE_TYPE_TO_VALID_ACTIONS = {}
