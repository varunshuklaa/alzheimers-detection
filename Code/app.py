from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import label_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_image(image):  
    text = label_image.main(image)
    return text

@app.route('/')
@app.route('/first')
def first():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # Ensure model file exists
        model_file_path = os.path.join(os.path.dirname(__file__), 'retrained_graph.pb')
        if not os.path.exists(model_file_path):
            return "Error: retrained_graph.pb file not found."

        # Prediction logic
        result = load_image(file_path)
        result = result.title()
        d = {
            "Verymild Demented": " → The Very Mild Demented (VMD) stage is an early phase of Alzheimer’s...",
            "Mild Demented": " → Mild Demented, also referred to as Mild Dementia due to Alzheimer's Disease...",
            "Moderate Demented": " → Moderate dementia refers to a middle stage of cognitive decline...",
            "Non Demented": " → Non-demented refers to individuals who do not exhibit significant cognitive decline..."
        }
        result = result + d.get(result, " → Description not found.")

        print(result)
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
