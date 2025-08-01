def get_schema_match_prompt(table_schema_dict, hint):

    return f"""
You are a highly meticulous and intelligent schema matcher specialized in data transformation tracing.

Your task is to analyze a set of **source tables** and a **target table**, and identify how each column in the target table semantically corresponds to column(s) in the source table(s).

The target table may be derived from the source tables using operations such as **join**, **union**, **groupby**, **pivot**, **unpivot**, **rename**, **column arithmetic**, **date formatting**, **adding or dropping columns**. 

You are provided with a high-level **hint** describing the likely transformation logic.

Each column in the target table must be mapped to its originating column(s) from the source tables. A target column may:
- Correspond to one or more columns from a single source table,
- Be the result of combining or aggregating columns,
- Be derived from columns across multiple source tables (e.g., through a join or union).

---

### Output Format

Your output should be a list of mappings, where each mapping is represented as a JSON object with the following keys:
- `"target_column"`: the name of a column in the target table.
- `"sources"`: an object mapping source table names to a **list** of the column(s) from each table that contribute to the target column.

Please respond with your reasoning process and a JSON object structured as follows:
# Reasoning Process:
(Here, output your reasoning for schema matching)
# Final JSON Result:
```json
[
  {{
    "target_column": "target_column_name1",
    "sources": {{
      "table_name1": ["column1", "column2", ...],
      "table_name2": ["column1", "column2", ...]
    }},
  }},
  {{
    "target_column": "target_column_name2",
    "sources": {{
      "table_name1": ["column1", "column2", ...],
      "table_name2": ["column1", "column2", ...]
    }},
  }},
]
```

Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.
Table Schema:\n{table_schema_dict}
Hint:\n{hint}

Please output your reasoning process first after "# Reasoning Process", and output a JSON object within ```json``` tags.
"""

def get_identify_function_prompt(table_schema_dict, hint):
    return f"""
You're an AI assistant that helps me identify functions that might be used from the source table to the target table.
Note that function can only be selected from the following: **join**, **union**, **groupby**, **pivot**, **unpivot**, **rename**, **column arithmetic**, **date formatting**, **adding or dropping columns**.

**************************
Example:
Source Tables:
**Table Caption:** test_0
**Columns:**
- EmployeeID
- Name
- DepartmentID
**Rows:**
1. | 1 | Alice | 101 |
2. | 2 | Bob | 102 |
**Table Caption:** test_1
**Columns:**
- DepartmentID
- DepartmentName
**Rows:**
1. | 103 | Finance |
2. | 104 | Marketing |
Target Table:
**Table Caption:** target
**Columns:**
- EmployeeID
- EmployeeName
- DepartmentName

Answer:
The transformation from the source tables to the target table likely involves the following functions:

Rename Columns:
Rename Name to EmployeeName in test_0.

Join Tables:
Perform a join between test_0 and test_1 on the column DepartmentID to add the DepartmentName to each employee.

Now, answer the real question, and you need to follow the answer style of the above examples
Table Schema:\n{table_schema_dict}

Hint:\n{hint}

Answer:
    """

def get_transformation_prompt(table_schema_dict, hint):
    return f"""
You are a data transformation expert specializing in table conversions using Python and pandas.

Your task is to write Python code that transforms one or more **source DataFrames** into a **target DataFrame**. The source tables are already loaded as pandas DataFrames with predefined variable names (e.g., `test_0`, `test_1`). Your solution must be **correct, clear, and reproducible**.

**Instructions:**

* Carefully analyze the structure of the source and target tables to determine the necessary transformation steps (e.g., `merge`, `groupby`, `pivot`, `melt`, `concat`, `rename`, `column arithmetic`, `date formatting`, `adding or dropping columns`).
* Always assume `merge` operations are inner joins (`how='inner'`) by default.
* When merging **multiple DataFrames sequentially (e.g., test_0 + test_1 + test_2 + test_3)** on the same key, be aware that:
  * **When generating code that merges multiple DataFrames, do not assume suffixes will appear unless a name conflict exists at that point.**
  * If the left table already has renamed columns (e.g., `col_test_0`), merging another table with `col` will not trigger a suffix.
* Provide a clear **step-by-step reasoning** outlining how to derive the target table from the source tables.
* Output **only** valid Python code using pandas that implements the transformation logic.
* Assign the final output DataFrame to the variable `target`.
* Do not include any additional explanation, comments, or markdown outside the required JSON structure.
* Each line in the code should be a complete python function and its arguments without line breaks
**Output Format:**
Respond with a JSON object in the following format:

```json
{{
  "chain_of_thought_reasoning": "Step-by-step reasoning explaining the transformation logic",
  "code": [
    "Python statement 1 (e.g., merged = pd.merge(test_0, test_2, on='zipcode', how='inner', suffixes=('_test_0', '_test_2')))",
    "Python statement 2 (e.g., merged = pd.merge(merged, test_3, on='zipcode', how='inner', suffixes=('', '')))",
    "Python statement 3 (e.g., merged = merged.rename(columns={{'businesses': 'businesses_test_3'}}))",
    "...",
    "target = final_transformed_dataframe[[target_column_1, target_column_2, ...]]"
  ]
}}
```

**Table Schema:**
{table_schema_dict}

**Hint:**
{hint}

Only output the JSON object above (starting with ```json and ending with ```), and nothing else.
"""

def get_transformation_revision_prompt(table_schema_dict, hint, original_code, error_message, exec_result):
    return f"""
You are a data transformation expert specializing in table conversions using Python and pandas.

A previous attempt to convert source tables into a target table failed — either due to a runtime error, unexpected output, or null results. Your task is to:

1. Analyze the provided table schema and the context of the failure.
2. Reason through what might have gone wrong.
3. Produce a corrected version of the Python code using pandas that successfully performs the transformation.
4. Each line in the code should be a complete python function and its arguments without line breaks

Your revised code must:

* Use the given source DataFrames (e.g., `test_0`, `test_1`) without renaming them.
* Always assume `merge` operations are inner joins (`how='inner'`) by default.
* Assign the final result to a variable named `target`.
* Be syntactically correct and logically aligned with the table schema and transformation goal.
* Focus on clarity, correctness, and reproducibility.

**Output Format:**
Respond with a JSON object enclosed in triple backticks:

```json
{{
  "chain_of_thought_reasoning": "Step-by-step reasoning identifying the error and explaining the fix",
  "code": [
    "Corrected Python statement 1",
    "Corrected Python statement 2",
    "...",
    "target = final_transformed_dataframe"
  ]
}}
```
**Original Code Attempt:**
{original_code}

**Original code execution result:**
{exec_result}

**Error Message:**
{error_message}

**Table Schema:**
{table_schema_dict}

**Hint:**
{hint}

Only output the JSON object above (starting with ```json and ending with ```), and nothing else.

    """
    
    
def get_reward_prompt(table_schema_dict, transformation, resulting_table, column_match_str):
    return f"""
You are a data transformation expert with exceptional knowledge of table processing using Python and pandas.

Your task is to evaluate whether the historical operations successfully transformed the **source tables** into the desired **target table structure**.

I will provide:

* The original source tables
* The target table
* The sequence of historical operations performed
* The resulting table after executing those operations

You must assess the transformation **based on structure and logic**, and determine whether a **rename operation is still required** and what the **correct reward score** should be.

---

### Evaluation Rules:
* Your answer should be a JSON object with:
  * `"reason"`: A brief explanation of the correctness or flaw in the historical operations.
  * `"reward"`: `"1"`, `"0.5"`, or `"0"` based on the criteria below.
---

### Reward Criteria:

| Reward  | Description                                                                                                                                                    |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**   | **Exact match**: All required transformations are complete and correct. The result table structure and logic match the target. |
| **0.5** | **Partially correct**: Some valid transformations are present, but more steps are needed (e.g., missing aggregation, join, or pivot).                          |
| **0**   | **Incorrect**: The logic of the operations is flawed, missing essential steps, or structurally incorrect.                                                      |

---

### Output Format:

Output a **JSON object enclosed in triple backticks** like this:

```json
{{
  "reason": "Concise explanation of correctness or flaw.",
  "reward": "1"
}}
```

---
**Table Schema:**
{table_schema_dict}

**Python code:**
{transformation}

**Resulting Table:**
{resulting_table}

**Hint:**
{column_match_str}

Only output the JSON object above (starting with ```json and ending with ```), and nothing else.
"""