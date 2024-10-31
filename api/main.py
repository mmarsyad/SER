import os
from flask import Flask, request, jsonify
from audio_predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return "Hi!"
    
@app.route('/api/predict/<filename>', methods=['POST'])
def predict_label(filename):
    if request.method == 'POST':
        sender = request.form['sender']
        path = 'path'
        folder_path = f'{path}/{sender}' # Access to WhatsApp API file
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 404, 'message': 'File not found'}), 404
        
        predicted_label, probabilty_label = predict(file_path)
        result = {}
        for i, x in enumerate(probabilty_label):
            emot = x[0]
            percentage = round(float(x[1]), 2)
            result[emot] = percentage
        
        return jsonify({'status': 200,
                        'data': [
                            {'filename': filename,
                             'predicted_label': predicted_label[0],
                             'probability': result}
                        ]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)