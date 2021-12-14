from cogdl.oag import oagbert


def test_encode_paper():
    tokenizer, model = oagbert("oagbert-v2-test")
    title = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation..."
    authors = ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
    venue = "north american chapter of the association for computational linguistics"
    affiliations = ["Google"]
    concepts = ["language model", "natural language inference", "question answering"]
    # encode paper
    paper_info = model.encode_paper(
        title=title,
        abstract=abstract,
        venue=venue,
        authors=authors,
        concepts=concepts,
        affiliations=affiliations,
        reduction="max",
    )

    assert len(paper_info) == 5
    assert paper_info["text"][0]["type"] == "TEXT"
    assert len(paper_info["authors"]) == 4
    assert len(paper_info["venue"][0]["token_ids"]) == 9
    assert tuple(paper_info["text"][0]["sequence_output"].shape) == (43, 768)
    assert len(paper_info["text"][0]["pooled_output"]) == 768


if __name__ == "__main__":
    test_encode_paper()
