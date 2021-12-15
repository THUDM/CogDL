import torch
from cogdl.oag import oagbert

tokenizer, bert_model = oagbert()
bert_model.eval()

sequence = ["CogDL is developed by KEG, Tsinghua.", "OAGBert is developed by KEG, Tsinghua."]
tokens = tokenizer(sequence, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = bert_model(**tokens)

print(outputs[0])
