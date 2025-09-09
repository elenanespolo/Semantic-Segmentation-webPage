from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
import io
from PIL import Image
from model.modelusage import segmented_image  
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/models')
def models():
    return render_template('Models.html')

@app.route('/aboutMe')
def aboutMe():
    return render_template('AboutMe.html')

@app.route("/legend", methods=["GET"])
def get_legend():
    legend = [
        {'name': 'road', 'color': [128, 64, 128]},
        {'name': 'sidewalk', 'color': [244, 35, 232]},
        {'name': 'building', 'color': [70, 70, 70]},
        {'name': 'wall', 'color': [102, 102, 156]},
        {'name': 'fence', 'color': [190, 153, 153]},
        {'name': 'pole', 'color': [153, 153, 153]},
        {'name': 'light', 'color': [250, 170, 30]},
        {'name': 'sign', 'color': [220, 220, 0]},
        {'name': 'vegetation', 'color': [107, 142, 35]},
        {'name': 'terrain', 'color': [152, 251, 152]},
        {'name': 'sky', 'color': [70, 130, 180]},
        {'name': 'person', 'color': [220, 20, 60]},
        {'name': 'rider', 'color': [255, 0, 0]},
        {'name': 'car', 'color': [0, 0, 142]},
        {'name': 'truck', 'color': [0, 0, 70]},
        {'name': 'bus', 'color': [0, 60, 100]},
        {'name': 'train', 'color': [0, 80, 100]},
        {'name': 'motocycle', 'color': [0, 0, 230]},
        {'name': 'bicycle', 'color': [119, 11, 32]},
        {'name': 'void', 'color': [0, 0, 0]}
    ]
    return jsonify(legend)

@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        img = Image.open(file.stream).convert('RGB')

        # Recupera il modello scelto (inviato dal frontend)
        model_name = request.form.get('model')
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400

        # Carica immagine
        img = Image.open(file.stream).convert('RGB')

        # Qui potresti scegliere il modello in base al valore
        if model_name == "DeepLabV2":
            img_segmented = segmented_image(img, model_name)
        elif model_name == "BiSeNet":
            img_segmented = segmented_image(img, model_name)
        elif model_name == "BiSeNetFDA":
            img_segmented = segmented_image(img, model_name)
        elif model_name == "BiSeNetDACS":
            img_segmented = segmented_image(img, model_name)
        else:
            raise ValueError("Unknown model selected")

        buf = io.BytesIO()
        img_segmented.save(buf, format="PNG")
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # usa la porta dinamica di Heroku
    app.run(host="0.0.0.0", port=port, debug=False)  # host 0.0.0.0, non 127.0.0.1
