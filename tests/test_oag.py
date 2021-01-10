from cogdl import oagbert


def test_oagbert():
    tokenizer, bert_model = oagbert("oagbert-test", load_weights=False)

    sequence = "CogDL is developed by KEG, Tsinghua."
    tokens = tokenizer(sequence, return_tensors="pt")
    outputs = bert_model(**tokens)

    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (1, 14, 32)
    assert tuple(outputs[1].shape) == (1, 32)


if __name__ == "__main__":
    test_oagbert()
