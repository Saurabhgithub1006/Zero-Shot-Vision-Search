# Singleton class to load SigLIP model and processor once.
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import threading

from src.config import get_settings
from src.device import select_device

def _features(output):
    """transformers 4.x returns a tensor, 5.x returns a model output holding it in pooler_output."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model and processor."""
        self.device = select_device()
        model_name = get_settings().embedding_model
        print(f"Loading SigLIP model on {self.device}...")

        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        print("Model loaded successfully.")

    def get_image_embedding(self, image_path):
        """
        Generate embedding for a single image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            list: The embedding vector as a list of floats.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = _features(self.model.get_image_features(**inputs))
                
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        """
        Generate embedding for a text query.
        
        Args:
            text (str): The text query.
            
        Returns:
            list: The embedding vector as a list of floats.
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding="max_length").to(self.device)
            
            with torch.no_grad():
                text_features = _features(self.model.get_text_features(**inputs))
                
            # Normalize the features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return None
