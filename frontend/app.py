from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    files = {'image': (image_file.filename, image_file, 'image/jpeg')}
    
    # Send request to API
    try:
        response = requests.post('http://localhost:5000/predict', files=files)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)