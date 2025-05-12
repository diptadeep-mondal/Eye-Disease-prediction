# app.py
from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import load_model

app = Flask(__name__)
model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                predicted = torch.argmax(output, dim=1).item()
                if predicted==0:
                    x='cataract'
                elif predicted==1:
                    x='diabetic_retinopathy'
                elif predicted==2:
                    x='glaucoma'
                else:
                    x='normal'
            return f"Predicted Class: {x}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

