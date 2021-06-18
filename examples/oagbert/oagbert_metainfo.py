from cogdl import oagbert
from cogdl.oag.utils import colored
import math

tokenizer, model = oagbert("oagbert-v2")
model.eval()

title = "Language Models are Few-Shot Learners"
abstract = """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general."""

# calculate the probability of `machine learning`, `artificial intelligence`, `language model` for GPT-3 paper
print("=== Span Probability ===")
for span in ["machine learning", "artificial intelligence", "language model"]:
    span_prob, token_probs = model.calculate_span_prob(
        title=title,
        abstract=abstract,
        decode_span_type="FOS",
        decode_span=span,
        mask_propmt_text="Field of Study:",
        debug=False,
    )
    print("%s probability: %.4f" % (span.ljust(30), span_prob))
print()

# decode a list of Field-Of-Study using beam search
concepts = []
print("=== Generated FOS ===")
for i in range(16):
    candidates = []
    for span_length in range(1, 5):
        results = model.decode_beamsearch(
            title=title,
            abstract=abstract,
            authors=[],
            concepts=concepts,
            decode_span_type="FOS",
            decode_span_length=span_length,
            beam_width=8,
            force_forward=False,
        )
        candidates.append(results[0])
    candidates.sort(key=lambda x: -x[1])
    span, prob = candidates[0]
    print(
        "%2d. %s %s" % (i + 1, span, colored("[%s]" % (",".join(["%s(%.4f)" % (k, v) for k, v in candidates])), "blue"))
    )
    concepts.append(span)
