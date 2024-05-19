import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import gdown
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Download the model file
model_url = 'https://drive.google.com/uc?id=1oZTHr7jIMG4R64iQKScwizE-Hw1KSh42'
output = 'model.h5'
gdown.download(model_url, output, quiet=False)

# Load the pre-trained model
try:
    model = load_model('model.h5')
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Define the class labels
labels = {0: 'dry_gangrene', 1: 'gas_gangrene', 2: 'normal_foot', 3: 'wet_gangrene'}

def preprocess_image(image_path, target_size=(225, 225)):
    img = image.load_img(image_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(image_path):
    if model:
        preprocessed_image = preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        predicted_label = labels[np.argmax(predictions[0])]
        return predicted_label
    else:
        return "Model not loaded"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    upload_folder = os.path.join(app.root_path, 'uploads')
    if not os.path.isdir(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, secure_filename(file.filename))
    file.save(file_path)

    predicted_label = predict_image(file_path)
    return predicted_label

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
