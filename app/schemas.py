"""Definitions for the objects used by our resource endpoints."""

from collections import namedtuple
from enum import Enum

from pydantic import BaseModel


class PredictPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 6.4,
                "sepal_width": 2.8,
                "petal_length": 5.6,
                "petal_width": 2.1,
            }
        }

class PredictBert(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "He is working as [MASK] in the university",
            }
        }

class PredictT5(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "translate English to German: Hello, how are you?",
            }
        }

class PredictCodeGen(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "def hello_world():",
            }
        }

class PredictPythia_70m(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "def hello_world():",
            }
        }

class PredictCodet5p_220m(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "def hello_world():<extra_id_0>",
            }
        }

class PredictCNN(BaseModel):
    input_text: str

    class Config:
        schema_extra = {
            "example": {
                "input_text": "10",
            }
        }

class IrisType(Enum):
    setosa = 0
    versicolor = 1
    virginica = 2