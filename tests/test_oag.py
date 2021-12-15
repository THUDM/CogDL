from cogdl.oag import oagbert


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
        title=sequence,
        decode_span_type="FOS",
        decode_span="data mining",
        mask_propmt_text="Field of Study:",
        debug=False,
    )
    assert span_prob >= 0 and span_prob <= 1
    results = model.decode_beamsearch(
        title=sequence, decode_span_type="FOS", decode_span_length=2, beam_width=2, force_forward=False
    )
    assert len(results) == 2
    model.generate_title(
        abstract="To enrich language models with domain knowledge is crucial but difficult. Based on the world's largest public academic graph Open Academic Graph (OAG), we pre-train an academic language model, namely OAG-BERT, which integrates massive heterogeneous entities including paper, author, concept, venue, and affiliation. To better endow OAG-BERT with the ability to capture entity information, we develop novel pre-training strategies including heterogeneous entity type embedding, entity-aware 2D positional encoding, and span-aware entity masking. For zero-shot inference, we design a special decoding strategy to allow OAG-BERT to generate entity names from scratch. We evaluate the OAG-BERT on various downstream academic tasks, including NLP benchmarks, zero-shot entity inference, heterogeneous graph link prediction, and author name disambiguation. Results demonstrate the effectiveness of the proposed pre-training approach to both comprehending academic texts and modeling knowledge from heterogeneous entities. OAG-BERT has been deployed to multiple real-world applications, such as reviewer recommendations for NSFC (National Nature Science Foundation of China) and paper tagging in the AMiner system. It is also available to the public through the CogDL package.",
        max_length=20,
    )


if __name__ == "__main__":
    test_oagbert()
    test_oagbert_v2()
