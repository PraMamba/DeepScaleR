"""Dataset type definitions for DeepScaler.

This module defines enums for training and testing datasets used in DeepScaler,
as well as a union type for both dataset types.
"""

import enum
from typing import Union


class TrainDataset(enum.Enum):
    """Enum for training datasets.
    
    Contains identifiers for various math problem datasets used during training.
    """
    AIME = 'AIME'  # American Invitational Mathematics Examination
    AMC = 'AMC'    # American Mathematics Competition
    OMNI_MATH = 'OMNI_MATH'  # Omni Math
    NUMINA_OLYMPIAD = 'OLYMPIAD'  # Unique Olympiad problems from NUMINA
    MATH = 'MATH'  # Dan Hendrycks Math Problems
    STILL = 'STILL'  # STILL dataset
    DEEPSCALER = 'DeepScaleR-Preview-Dataset'  # DeepScaler (AIME, AMC, OMNI_MATH, MATH, STILL)


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.
    
    Contains identifiers for datasets used to evaluate model performance.
    """
    AIME_24 = 'AIME-2024'  # American Invitational Mathematics Examination
    AMC = 'AMC_2022-2023'    # American Mathematics Competition  
    MATH = 'MATH-500'  # Math 500 problems
    MINERVA = 'MINERVA'  # Minerva dataset
    OLYMPIAD_BENCH = 'OLYMPIAD_BENCH'  # Olympiad benchmark problems
    AIME_25 = 'AIME-2025'
    HMMT_202502 = 'HMMT-202502'
    AIMO2_Reference = 'AIMO-2_Reference'

"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]
