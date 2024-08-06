from flask import Flask, request, jsonify
from huggingface.fine_tune import HuggingFaceFineTune
import os
import logging

app = Flask(__name__)
hf_fine_tune = HuggingFaceFineTune()

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/choosePlatform', methods=['POST'])
def choose_platform():
    """
    Endpoint to choose the platform.
    """
    platform = request.json.get('platform')
    try:
        response = hf_fine_tune.choose_platform(platform)
        return jsonify(response), 200
    except ValueError as e:
        logging.error(f"Error in /choosePlatform: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/provideToken', methods=['POST'])
def provide_token():
    """
    Endpoint to provide the authentication token.
    """
    token = request.json.get('token')
    try:
        response = hf_fine_tune.provide_token(token)
        return jsonify(response), 200
    except ValueError as e:
        logging.error(f"Error in /provideToken: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/selectModel', methods=['POST'])
def select_model():
    """
    Endpoint to select the model.
    """
    model_name = request.json.get('model_name')
    try:
        response = hf_fine_tune.select_model(model_name)
        return jsonify(response), 200
    except ValueError as e:
        logging.error(f"Error in /selectModel: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/uploadData', methods=['POST'])
def upload_data():
    """
    Endpoint to upload dataset for training.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            response = hf_fine_tune.upload_data(file_path)
            return jsonify(response), 200
        except ValueError as e:
            logging.error(f"Error in /uploadData: {e}")
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Invalid file format. Only CSV files are supported."}), 400

@app.route('/configureHyperparameters', methods=['POST'])
def configure_hyperparameters():
    """
    Endpoint to configure hyperparameters for fine-tuning.
    """
    data = request.json
    hyperparameters = data.get('hyperparameters', {})
    if all(k in hyperparameters for k in ('r', 'lora_alpha', 'lora_dropout')):
        try:
            hf_fine_tune.configure_hyperparameters(hyperparameters)
            return jsonify({"message": "Hyperparameters configured successfully."}), 200
        except Exception as e:
            logging.error(f"Error in /configureHyperparameters: {e}")
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "All hyperparameters are required."}), 400

@app.route('/fineTune', methods=['POST'])
def fine_tune():
    """
    Endpoint to initiate fine-tuning.
    """
    data = request.json
    output_dir = data.get('output_dir')
    hub_model_id = data.get('hub_model_id')
    learning_rate = data.get('learning_rate')
    num_train_epochs = data.get('num_train_epochs')

    if all(v is not None for v in [output_dir, hub_model_id, learning_rate, num_train_epochs]):
        try:
            result_model_id = hf_fine_tune.fine_tune(output_dir, hub_model_id, learning_rate, num_train_epochs)
            return jsonify({"message": "Fine-tuning initiated.", "model_id": result_model_id}), 200
        except Exception as e:
            logging.error(f"Error in /fineTune: {e}")
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "All fields are required for fine-tuning."}), 400

@app.route('/compareResults', methods=['GET'])
def compare_results():
    """
    Endpoint to compare results of the fine-tuned model with the base model.
    """
    hub_model_id = request.args.get('hub_model_id')
    if hub_model_id:
        try:
            results = hf_fine_tune.compare_results(hub_model_id)
            return jsonify(results), 200
        except Exception as e:
            logging.error(f"Error in /compareResults: {e}")
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Model ID is required for comparison."}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Ensure upload directory exists
    app.run(debug=True)
