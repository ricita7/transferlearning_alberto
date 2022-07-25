import requests
from transformers import BertTokenizer, BertForPreTraining
import torch
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
model = BertForPreTraining.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

data = pd.read_csv('dataset2020.csv', encoding='utf-8', sep = ';')
data.dropna(inplace=True)


text = []
for sentence in data['testo']:
  text.append(sentence)

inputs = tokenizer(text[:9000], return_tensors = 'pt', max_length = 512, truncation = True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)

# We mask 15% of the tokens without including the special tokens

mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

for i in range(inputs.input_ids.shape[0]):
  selection = torch.flatten(mask_arr[i].nonzero()).tolist()
  inputs.input_ids[i, selection] = 103 #masking token

class IstatDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

dataset = IstatDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle = True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

from transformers import AdamW
#l = torch.nn.CrossEntropyLoss()
optim = AdamW(model.parameters(), lr = 5e-5)

model.train()
from tqdm import tqdm

for epoch in range(2):
  loop = tqdm(loader, leave = True)
  for batch in loop:
    optim.zero_grad()
    # tensors to device
    input_ids = batch['input_ids'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=attention_mask, labels=labels)
    
    #print(labels.shape)
    #print(outputs.prediction_logits.shape)
    #print(outputs.prediction_logits)

    loss = outputs.loss
    loss.backward()
    optim.step()
    
    loop.set_description(f'Epoch {epoch}')
    loop.set_postfix(loss=loss.item())
