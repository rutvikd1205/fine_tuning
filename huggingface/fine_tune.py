import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import torch
from datasets import Dataset

print(torch.backends.mps.is_available())

class HuggingFaceFineTune:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.dataset = None
        self.device = torch.device("mps")
        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     print("mps")
        # elif torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")
        self.platform = None
        self.token = None
        self.model_name = None

    def choose_platform(self, platform):
        if platform == 'HuggingFace':
            self.platform = platform
            return {"message": "HuggingFace platform selected."}
        else:
            raise ValueError("Invalid platform selected.")

    def provide_token(self, token):
        from huggingface_hub import login
        login(token)
        self.token = token
        return {"message": "Token provided successfully."}

    def select_model(self, model_name):
        if not self.platform or not self.token:
            raise ValueError("Platform or token not set. Please set both before selecting a model.")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("mps")
        print(f"Model device: {self.model.device}")
        return {"message": "Model selected successfully."}

    def upload_data(self, csv_file):
        if not self.model_name:
            raise ValueError("Model not selected. Please select a model before uploading data.")
        
        # Load CSV file into DataFrame
        df = pd.read_csv(csv_file)

        # Split the DataFrame into train, test, and validation
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Convert DataFrames to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        self.dataset = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

        # Check dataset size
        if len(self.dataset['train']) < 500:
            raise ValueError("Dataset must contain more than 500 samples.")

        self._preprocess_data()
        return {"message": "Dataset uploaded and processed successfully."}

    def _preprocess_data(self):
        max_length = 256

        def tokenize_inputs(example):
            start_prompt = "Summarize the following conversation. \n\n"
            end_prompt = "\n\nSummary :"
            prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
            inputs = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            labels = self.tokenizer(example['summary'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

            example['input_ids'] = inputs['input_ids'].tolist()
            example['labels'] = labels['input_ids'].tolist()

            return example

        tokenized_datasets = self.dataset['train'].map(tokenize_inputs, batched=True)
        val_tokenized_datasets = self.dataset['validation'].map(tokenize_inputs, batched=True)
        test_tokenized_datasets = self.dataset['test'].map(tokenize_inputs, batched=True)

        self.dataset['train'] = tokenized_datasets.remove_columns(['id', 'dialogue', 'summary'])
        self.dataset['validation'] = val_tokenized_datasets.remove_columns(['id', 'dialogue', 'summary'])
        # self.dataset['test'] = test_tokenized_datasets.remove_columns(['id', 'dialogue', 'summary'])

    def configure_hyperparameters(self, hyperparameters):
        lora_config = LoraConfig(
            r=hyperparameters['r'],
            lora_alpha=hyperparameters['lora_alpha'],
            lora_dropout=hyperparameters['lora_dropout'],
            bias='none',
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.peft_model = get_peft_model(self.model, peft_config=lora_config).to("mps")
        print(f"PEFT model device: {self.peft_model.device}")
        print(f"PEFT config: {lora_config}")

    def fine_tune(self, output_dir, hub_model_id, learning_rate, num_train_epochs):
        if self.peft_model is None:
            raise ValueError("PEFT model has not been configured. Please configure hyperparameters before fine-tuning.")
        
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            hub_model_id=hub_model_id,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            per_device_train_batch_size=2,
            evaluation_strategy="epoch", 
            logging_steps=10
        )

        trainer = Trainer(
            model=self.peft_model.to("mps"),
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation']
        )

        def _move_inputs_to_device(inputs):
            print(f"Input device: {next(iter(inputs.values())).device}")
            return {key: value.to("mps") for key, value in inputs.items()}
        

        # Overriding the default compute_loss to move inputs to device
        # trainer.compute_loss = lambda model, inputs: model(**_move_inputs_to_device(inputs)).loss
        
        def compute_loss(model, inputs, return_outputs=False):
            inputs = _move_inputs_to_device(inputs)
            outputs = model(**inputs)
            loss = outputs.loss.to("cpu")  # Move the loss to CPU
            return (loss, outputs) if return_outputs else loss

        trainer.compute_loss = compute_loss

        self.peft_model.print_trainable_parameters()
        trainer.train()
        trainer.push_to_hub()

        return hub_model_id

    def compare_results(self, hub_model_id):
        config = PeftConfig.from_pretrained(hub_model_id)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to("mps")  # Use the selected model name
        loaded_peft_model = PeftModel.from_pretrained(
            self.model,
            hub_model_id,
            is_trainable=False
        ).to("mps")

        sample = self.dataset['test'][0]['dialogue']
        label = self.dataset['test'][0]['summary']

        def generate_summary(input_text, model):
            model.eval()
            with torch.no_grad():
                input_prompt = f"""
                                Summarize the following conversation.

                                {input_text}

                                Summary:
                                """
                inputs = self.tokenizer(input_prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                tokenized_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_length=40,
                    max_length=200
                )

                output = self.tokenizer.decode(tokenized_output[0], skip_special_tokens=True)
            return output

        output_peft = generate_summary(sample, model=loaded_peft_model)
        output_original = generate_summary(sample, model=base_model)

        return {
            "PEFT_model_output": output_peft,
            "Original_model_output": output_original
        }
