# __init__.py

from .ai_model import AIModel
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .inference import InferenceEngine
from .utils import load_config

# Initialize components
def setup_ai_engine():
    load_config()  # Load any global configurations for AI Engine
    print("AI Engine setup completed.")
