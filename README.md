# Fine-Tuning Application API

These are the 7 REST endpoints:

1. **choosePlatform:** To select the platform for fine-tuning (HuggingFace or Lamini).
2. **provideToken:** To provide a writeable access token for the selected platform.
3. **uploadData:** To upload the dataset for fine-tuning.
4. **selectModel:** To choose the model to be fine-tuned.
5. **configureHyperparameters:** To set the hyperparameters for LoRA fine-tuning.
6. **fineTune:** To initiate the fine-tuning process.
7. **compareResults:** To compare the performance of the fine-tuned model against the original model.

# STEPS TO TEST THE FINETUNING PLATFORM
# HuggingFace Model Fine-Tuning API

This API allows you to fine-tune HuggingFace models using CSV file uploads, configure hyperparameters, and compare results.

## Usage

To use the API, execute the following curl commands sequentially:

```sh
# Choose Platform (Currently "HuggingFace")
curl -X POST http://127.0.0.1:5000/choosePlatform \
-H "Content-Type: application/json" \
-d '{"platform": "HuggingFace"}'

# Provide Token
curl -X POST http://127.0.0.1:5000/provideToken \
-H "Content-Type: application/json" \
-d '{"token": "your_access_token_with_write_permission"}'

# Select Model (Either of the following: 1) google-t5/t5-base, 2)meta-llama/Meta-Llama-3.1-8B-Instruct)
curl -X POST http://127.0.0.1:5000/selectModel \
-H "Content-Type: application/json" \
-d '{"model_name": "google-t5/t5-base"}'

# Upload Data
curl -X POST http://127.0.0.1:5000/uploadData \
-F "file=@/path/to/your/file.csv"

# Configure Hyperparameters
curl -X POST "http://127.0.0.1:5000/configureHyperparameters" -H "Content-Type: application/json" -d '{
    "hyperparameters": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1
    }
}'

# Fine-Tune Model (Create a new repo on hugging face. Eg; 'rutvikd0512/modular_test')
curl -X POST http://127.0.0.1:5000/fineTune \
-H "Content-Type: application/json" \
-d '{
      "output_dir": "path_to_your_local_folder",
      "hub_model_id": "path_to_the_huggingface_repo",
      "learning_rate": 5e-5,
      "num_train_epochs": 3
    }'

# Compare Results
curl -X GET "http://127.0.0.1:5000/compareResults?hub_model_id=path_to_the_huggingface_repo" \
-H "Content-Type: application/json"

