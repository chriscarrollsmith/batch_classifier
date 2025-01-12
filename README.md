# LLM Classifier

This project classifies text data from a CSV file using the OpenAI batch API. The batch API allows large workloads to be run at off-peak times at a fraction of the cost of running the same workload on the regular OpenAI API.

## Prerequisites

Before you begin, make sure you have the following installed:

1.  **uv**: This project uses `uv` for package management. Install it using the following command:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    See the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more details.

2.  **Python**: Install Python using `uv`:
    ```bash
    uv python install
    ```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/chriscarrollsmith/batch-classifier.git
    cd batch-classifier
    ```

2.  Install the project dependencies using `uv`:
    ```bash
    uv sync
    ```

3.  Create a `.env` file in the root directory and add your [OpenAI API key](https://platform.openai.com/api-keys):
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1.  Prepare your input CSV file named `input.csv` and define your `prompt_template` and `ClassificationResponse` model in `prompt.py`. Ensure that the column names in your CSV match the placeholders in `prompt_template`. For example, if your prompt uses `{item}`, your CSV should have a column named `item`.

    Note: The classifier supports Pydantic models with nested structures (which will be flattened in the output CSV) and Enum fields.

2.  Run the `submit_batch.py` script:
    ```bash
    uv run python submit_batch.py
    ```

    This script will read `input.csv` and construct a batch request to classify each item using the LLM, and output helper files called `batch_info.json` and `batch_results.jsonl` to track the batch request and process the results.

3.  Run the `process_batch.py` script:
    ```bash
    uv run python process_batch.py
    ```
    This script will read `batch_info.json` and `batch_results.jsonl`, make a request to retrieve the batch results, and process the returned results, outputting the results to `output.csv`. If the batch request is still in progress, the script will wait for the batch request to complete (polling every 30 seconds) and then process the results.

If your output CSV contains rows that already have complete observations, these rows will be preserved unchanged when using the output as input for another run. This allows you to process missed items by feeding the output CSV back in as input.