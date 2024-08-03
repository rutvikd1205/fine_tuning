# Fine-Tuning Application API

These are the 7 REST endpoints:

1. **choosePlatform:** To select the platform for fine-tuning (HuggingFace or Lamini).
2. **provideToken:** To provide a writeable access token for the selected platform.
3. **uploadData:** To upload the dataset for fine-tuning.
4. **selectModel:** To choose the model to be fine-tuned.
5. **configureHyperparameters:** To set the hyperparameters for LoRA fine-tuning.
6. **fineTune:** To initiate the fine-tuning process.
7. **compareResults:** To compare the performance of the fine-tuned model against the original model.


# Sequential REST Endpoint Calls Workflow

## choosePlatform
User selects the platform.

Upon success, the application proceeds to the next step.

## provideToken
User provides the writeable access token.

Upon success, the application proceeds to the next step.

## uploadData
User uploads their dataset.

If the data is sufficient, the application proceeds to the next step. If not, prompt the user to upload a larger dataset.

## selectModel
User selects the model to be fine-tuned.

Upon success, the application proceeds to the next step.

## configureHyperparameters
User sets the hyperparameters for fine-tuning.

Upon success, the application proceeds to the next step.

## fineTune
The fine-tuning process is initiated.

The application provides a job ID to track the fine-tuning process.

## compareResults
The application fetches and compares the results using the job ID.
