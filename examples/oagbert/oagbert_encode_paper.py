from cogdl.oag import oagbert

tokenizer, model = oagbert("oagbert-v2")
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

for name, content in paper_info.items():
    print(name)
    print(content)
