import json
import time
import dotenv
import pandas as pd
from litellm import aretrieve_batch, Batch, afile_content, HttpxBinaryResponseContent
import asyncio
from classifier import (
    parse_llm_json_response,
    update_classifications,
    prepare_dataframe,
    get_empty_mask,
)
import os

dotenv.load_dotenv()

async def process_batch_results(batch_info_file: str = "batch_info.json") -> None:
    """Process the results of a batch classification request."""
    # Load batch information
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)
    
    # Retrieve batch results
    while True:
        batch_response: Batch = await aretrieve_batch(
            batch_id=batch_info['batch_id'],
            custom_llm_provider="openai"
        )
        print(f"Batch status: {batch_response.status}")
        
        if batch_response.status == "completed":
            results = []
            if batch_response.output_file_id:
                # Get successful results from the output file
                output_content: HttpxBinaryResponseContent = await afile_content(
                    file_id=batch_response.output_file_id,
                    custom_llm_provider="openai"
                )
                for line in output_content.text.splitlines():
                    if line.strip():
                        results.append(json.loads(line))
            
            # Get any error results
            if batch_response.error_file_id:
                error_content: HttpxBinaryResponseContent = await afile_content(
                    file_id=batch_response.error_file_id,
                    custom_llm_provider="openai"
                )
                print("\nProcessing errors:")
                for line in error_content.text.splitlines():
                    if line.strip():
                        error_json = json.loads(line)
                        error_msg = error_json['response']['body']['error']['message']
                        print(f"Request {error_json['custom_id']} failed: {error_msg}")
                
            break
        elif batch_response.status == "failed":
            if batch_response.error_file_id:
                error_content: HttpxBinaryResponseContent = await afile_content(
                    file_id=batch_response.error_file_id,
                    custom_llm_provider="openai"
                )
                error_details = []
                for line in error_content.text.splitlines():
                    if line.strip():
                        error_json = json.loads(line)
                        error_msg = error_json['response']['body']['error']['message']
                        error_details.append(f"Request {error_json['custom_id']}: {error_msg}")
                raise Exception(f"Batch processing failed.\nErrors:\n" + "\n".join(error_details))
            else:
                raise Exception("Batch processing failed with no error details")
        
        print("Waiting 30 seconds...")
        time.sleep(30)
    
    # Process the results
    df = pd.read_csv(batch_info['input_file'])
    model_fields = batch_info['model_fields']
    df = prepare_dataframe(df, model_fields)
    mask = get_empty_mask(df, model_fields)
    
    # Parse the results
    parsed_results = []
    for result in results:
        try:
            print(result)
            parsed = parse_llm_json_response(
                result['response']['body']['choices'][0]['message']['content'],
                ClassificationResponse
            )
            parsed_results.append(parsed)
        except Exception as e:
            print(f"Error parsing result: {str(e)}")
            parsed_results.append(None)
    
    # Update the DataFrame with new classifications
    df = update_classifications(df, parsed_results, mask, model_fields)
    
    # Save the updated DataFrame
    df.to_csv(batch_info['output_file'], index=False)
    print(f"Results saved to {batch_info['output_file']}")
    
    # Clean up temporary files only after successful processing
    os.remove(batch_info_file)
    os.remove(batch_info.get('input_jsonl', 'batch_requests.jsonl'))
    print("Cleaned up temporary files")

if __name__ == "__main__":
    from prompt import ClassificationResponse
    
    asyncio.run(process_batch_results()) 