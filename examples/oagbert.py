from cogdl.oag import oagbert

tokenizer, bert_model = oagbert()

sequence = "CogDL is developed by KEG, Tsinghua."
tokens = tokenizer(sequence, return_tensors="pt")
outputs = bert_model(**tokens)

print(outputs[0].shape)
