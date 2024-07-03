from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Example sentences (dummy data)
sentences = [
    "The transformer is operating at [MASK] temperature.",
    "The substation voltage is [MASK] kV.",
]

# Tokenize sentences
inputs = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Define labels
labels = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True, padding='max_length')['input_ids']

# Masked language model training
class CPPSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = CPPSDataset(inputs, labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    save_steps=10,                   # save checkpoint every 10 steps
    save_total_limit=2,              # save only the last 2 checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('./cpps_bert')
tokenizer.save_pretrained('./cpps_bert')