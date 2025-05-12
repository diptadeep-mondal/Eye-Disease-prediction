# model.py
import torch
from your_model_file import animal_classification  # Replace with actual class

def load_model():
    model = animal_classification()
    model.load_state_dict(torch.load("models/eye_disease_model_state.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
