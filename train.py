import json

from transformers import EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer, BertConfig, Trainer, TrainingArguments
from datasets import Dataset

# Load your data
with open('training_data/mappings_size_5.json') as f:
    mappings = json.load(f)

inputs = [key for key in mappings.keys()]
outputs = [value for value in mappings.values()]

# Convert data to the correct format for Dataset.from_dict
data = {"input": inputs, "output": outputs}
dataset = Dataset.from_dict(data)

# Split dataset into training and validation
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def process_data_to_model_inputs(batch):
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

# Process data
train_dataset = train_dataset.map(process_data_to_model_inputs, batched=True)
test_dataset = test_dataset.map(process_data_to_model_inputs, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])

# Create configurations
encoder_config = BertConfig(vocab_size=tokenizer.vocab_size, is_decoder=False, hidden_size=768, num_hidden_layers=6)
decoder_config = BertConfig(vocab_size=tokenizer.vocab_size, is_decoder=True, hidden_size=768, num_hidden_layers=6, add_cross_attention=True)

# Create an EncoderDecoderConfig
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

# Initialize model from configuration
model = EncoderDecoderModel(config=config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=4,  # Adjust batch size according to your GPU memory
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,  # Adjust epochs according to your dataset size
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
