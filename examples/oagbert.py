from cogdl import oagbert

tokenizer, bert_model = oagbert()

sequence = ["CogDL is developed by KEG, Tsinghua.", "OAGBert is developed by KEG, Tsinghua."]
tokens = tokenizer(sequence, return_tensors="pt", padding=True)
outputs = bert_model(**tokens)

print(outputs[0].shape)
