import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Import your model class
from model import eye_classification  # Make sure this class is defined in model.py

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define class labels (update as needed)
class_names = ['Cataract','Retinopathy', 'Glaucoma','Healthy']

# Load model
model = eye_classification()  # Initialize the model
model.load_state_dict(torch.load('models/eye_disease_model_state.pth', map_location='cpu', weights_only=False))  # Load weights
model.eval()  # Set model to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),  # Match training image size
    transforms.ToTensor()
])

def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Transform and predict
    img_tensor = transform_image(filepath)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        result = class_names[predicted.item()]

    return render_template('result.html', result=result, image_url=url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
