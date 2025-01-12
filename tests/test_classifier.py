import pytest
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from classifier import parse_llm_json_response


# Test enums
class PrimaryClassification(str, Enum):
    TYPE_A = "type_a"
    TYPE_B = "type_b"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ProjectType(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"

# Test nested model
class ClassificationDetails(BaseModel):
    primary: PrimaryClassification
    confidence: ConfidenceLevel
    project_type: ProjectType

class NestedClassificationResponse(BaseModel):
    classification: ClassificationDetails
    justification: str
    evidence: str


class TestResponse(BaseModel):
    reason: str
    classification: str


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'item': ['test item'],
        'category': ['test category']
    })


@pytest.fixture
def test_files(tmp_path):
    input_file = tmp_path / "test_input.csv"
    output_file = tmp_path / "test_output.csv"
    return input_file, output_file


def test_parse_llm_json_response_direct_json():
    """Test parsing direct JSON response"""
    content = '{"reason": "test reason", "classification": "person"}'
    result = parse_llm_json_response(content, TestResponse)
    assert result.reason == "test reason"
    assert result.classification == "person"


def test_parse_llm_json_response_markdown_json():
    """Test parsing JSON within markdown code fence"""
    content = '```json\n{"reason": "test reason", "classification": "place"}\n```'
    result = parse_llm_json_response(content, TestResponse)
    assert result.reason == "test reason"
    assert result.classification == "place"

