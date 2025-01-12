import json
import dotenv
import pandas as pd
from pydantic import BaseModel
from typing import Type, TypeVar, List, Dict

import nest_asyncio

from enum import Enum

dotenv.load_dotenv()
nest_asyncio.apply()

# Type variable for generic JSON parsing
T = TypeVar('T', bound=BaseModel)


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code fence if present."""
    if '```json' in content:
        return content.split('```json')[1].split('```')[0].strip()
    return content.strip().strip('"\'')


def parse_llm_json_response(content: str, model_class: Type[T]) -> T:
    """Parse JSON from LLM response, handling both direct JSON and markdown-fenced output."""
    try:
        return model_class.model_validate(json.loads(content))
    except json.JSONDecodeError:
        json_str = extract_json_from_markdown(content)
        return model_class.model_validate(json.loads(json_str))


def get_format_args(row: pd.Series, placeholders: List[str]) -> Dict[str, str]:
    """Extract format arguments from a row based on placeholders."""
    format_args = {}
    for col in placeholders:
        if col not in row:
            raise TemplateError(f"Column '{col}' in prompt template not found in CSV file.")
        format_args[col] = row.get(col, '')
    return format_args


class ClassifierError(Exception):
    """Base exception for classifier errors"""
    pass


class TemplateError(ClassifierError):
    """Raised when there's an error with the prompt template"""
    pass


class APIError(ClassifierError):
    """Raised when there's an error with the LLM API"""
    pass


def prepare_dataframe(df: pd.DataFrame, model_fields: List[str]) -> pd.DataFrame:
    """Prepare DataFrame by ensuring all required columns exist with string dtype."""
    for field in model_fields:
        if field not in df.columns:
            df[field] = ''
        # Ensure all fields are string type
        df[field] = df[field].astype(str).replace('nan', '')
    return df


def get_empty_mask(df: pd.DataFrame, model_fields: List[str]) -> pd.Series:
    """Create a mask for rows that need classification."""
    mask = pd.Series(False, index=df.index)
    for field in model_fields:
        # Handle NaN values first
        field_mask = df[field].isna()
        
        # Handle empty strings and 'NA' strings for non-NaN values
        non_nan_mask = ~df[field].isna()
        if non_nan_mask.any():
            str_values = df.loc[non_nan_mask, field].astype(str)
            empty_str_mask = pd.Series(False, index=df.index)
            str_mask = (str_values == '') | (str_values.str.lower() == 'na')
            empty_str_mask.loc[non_nan_mask] = str_mask.astype(bool)
            field_mask = field_mask | empty_str_mask
        
        mask = mask | field_mask
    return mask


def flatten_model_dict(model_dict: Dict) -> Dict:
    """Recursively flatten a nested dictionary from a Pydantic model."""
    flattened = {}
    for key, value in model_dict.items():
        if isinstance(value, dict):
            nested = flatten_model_dict(value)
            flattened.update(nested)
        else:
            flattened[key] = value
    return flattened


def update_classifications(df: pd.DataFrame, results: List[T | None], mask: pd.Series, model_fields: List[str]) -> pd.DataFrame:
    """Update DataFrame with new classifications."""
    if not any(mask) or not results:  # No rows to update or no results
        return df
    
    # Filter out None results while keeping track of indices
    valid_results = [(i, result) for i, result in enumerate(results) if result is not None]
    if not valid_results:  # No valid results
        return df
    
    indices, filtered_results = zip(*valid_results)
    result_dicts = [flatten_model_dict(result.model_dump()) for result in filtered_results]
    result_df = pd.DataFrame(result_dicts)
    
    # Get the indices of rows that had successful classifications
    mask_indices = df[mask].index
    successful_indices = [mask_indices[i] for i in indices]
    result_df.index = successful_indices
    
    for field in model_fields:
        # Convert values to strings before assignment
        if isinstance(result_df[field].iloc[0], Enum):
            values = result_df[field].apply(lambda x: x.value)
        else:
            values = result_df[field].astype(str)
        # Ensure the column is string type before assignment
        df[field] = df[field].astype(str)
        df.loc[successful_indices, field] = values
    
    return df


def flatten_model_fields(model_class: Type[BaseModel]) -> List[str]:
    """Recursively flatten Pydantic model fields to get leaf field names."""
    fields = []
    for field_name, field in model_class.model_fields.items():
        if hasattr(field.annotation, 'model_fields'):  # If field is another Pydantic model
            nested_fields = flatten_model_fields(field.annotation)
            fields.extend(nested_fields)
        else:
            fields.append(field_name)
    return fields
