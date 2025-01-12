import re
import json
import dotenv
from pydantic import BaseModel
import pandas as pd
from litellm import acreate_file, acreate_batch
from typing import Type, TypeVar, Dict
import asyncio
from classifier import (
    get_format_args,
    prepare_dataframe,
    get_empty_mask,
    flatten_model_fields
)

dotenv.load_dotenv()

T = TypeVar('T', bound='BaseModel')

async def create_batch_jsonl(
    df: pd.DataFrame,
    prompt_template: str,
    output_file: str = "batch_requests.jsonl"
) -> str:
    """Create a JSONL file for batch processing."""
    placeholders = re.findall(r'\{(\w+)\}', prompt_template)
    
    # Validate template placeholders before processing
    format_args = get_format_args(df.iloc[0], placeholders)
    
    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            format_args = get_format_args(row, placeholders)
            current_prompt = prompt_template.format(**format_args)
            request = {
                "custom_id": f"row_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": current_prompt}],
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + '\n')
    
    return output_file

async def submit_batch_request(
    input_file: str,
    output_file: str,
    prompt_template: str,
    model_class: Type[T]
) -> Dict:
    """Submit a batch classification request."""
    # Read and prepare the DataFrame
    df = pd.read_csv(input_file)
    model_fields = flatten_model_fields(model_class)
    df = prepare_dataframe(df, model_fields)
    
    # Identify rows needing classification
    mask = get_empty_mask(df, model_fields)
    rows_to_classify = df[mask].copy()
    
    if rows_to_classify.empty:
        print("No rows need classification")
        return None
    
    # Create JSONL file for batch processing
    jsonl_file = await create_batch_jsonl(rows_to_classify, prompt_template)
    
    # Submit the file for batch processing
    file_obj = await acreate_file(
        file=open(jsonl_file, "rb"),
        purpose="batch",
        custom_llm_provider="openai"
    )
    
    # Create the batch request
    batch_response = await acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="openai",
        metadata={"input_file": input_file, "output_file": output_file}
    )
    
    # Save batch information for later processing
    batch_info = {
        "batch_id": batch_response.id,
        "file_id": file_obj.id,
        "input_file": input_file,
        "output_file": output_file,
        "rows_to_classify": len(rows_to_classify),
        "model_fields": model_fields
    }
    
    with open("batch_info.json", "w") as f:
        json.dump(batch_info, f)
    
    return batch_info

if __name__ == "__main__":
    from prompt import prompt_template, ClassificationResponse
    
    input_csv = "input.csv"
    output_csv = "output.csv"
    
    batch_info = asyncio.run(submit_batch_request(
        input_csv,
        output_csv,
        prompt_template,
        ClassificationResponse
    ))
    
    if batch_info:
        print(f"Batch request submitted successfully. Batch ID: {batch_info['batch_id']}")
        print("Run process_batch.py to retrieve and process the results when ready.") 