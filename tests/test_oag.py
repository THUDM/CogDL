from cogdl import oagbert


def test_oagbert():
    tokenizer, bert_model = oagbert("oagbert-test", load_weights=False)

    sequence = "CogDL is developed by KEG, Tsinghua."
    tokens = tokenizer(sequence, return_tensors="pt")
    outputs = bert_model(**tokens, checkpoint_activations=True)

    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (1, 14, 32)
    assert tuple(outputs[1].shape) == (1, 32)


def test_oagbert_v2():
    tokenizer, model = oagbert("oagbert-v2-test")
    sequence = "CogDL is developed by KEG, Tsinghua."
    span_prob, token_probs = model.calculate_span_prob(
        title=sequence, decode_span_type='FOS', decode_span='data mining', mask_propmt_text='Field of Study:', debug=False)
    assert span_prob >= 0 and span_prob <= 1
    results = model.decode_beamsearch(title=sequence, decode_span_type='FOS', decode_span_length=2, beam_width=2, force_forward=False)
    assert len(results) == 2

if __name__ == "__main__":
    test_oagbert()
    test_oagbert_v2()
