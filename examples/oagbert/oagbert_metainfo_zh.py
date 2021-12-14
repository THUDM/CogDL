from cogdl.oag import oagbert
from cogdl.oag.utils import colored
import math

tokenizer, model = oagbert("oagbert-v2-zh")
model.eval()

title = "基于随机化矩阵分解的网络嵌入方法"
abstract = """随着互联网的普及,越来越多的问题以社交网络这样的网络形式出现.网络通常用图数据表示,由于图数据处理的挑战性,如何从图中学习到重要的信息是当前被广泛关注的问题.网络嵌入就是通过分析图数据得到反映网络结构的特征向量,利用它们进而实现各种数据挖掘任务,例如边预测、节点分类、网络重构、标签推荐和异常检测.最近,基于矩阵分解的网络嵌入方法NetMF被提出,它在理论上统一了多种网络嵌入方法,并且在处理实际数据时表现出很好的效果.然而,在处理大规模网络时,NetMF需要极大的时间和空间开销.本文使用快速随机化特征值分解和单遍历奇异值分解技术对NetMF进行改进,提出一种高效率、且内存用量小的矩阵分解网络嵌入算法eNetMF.首先,我们提出了适合于对称稀疏矩阵的随机化特征值分解算法freigs,它在处理实际的归一化网络矩阵时比传统的截断特征值分解算法快近10倍,且几乎不损失准确度.其次,我们提出使用单遍历奇异值分解处理NetMF方法中高次近似矩阵从而避免稠密矩阵存储的技术,它大大减少了网络嵌入所需的内存用量.最后,我们提出一种简洁的、且保证分解结果对称的随机化单遍历奇异值分解算法,将它与上述技术结合得到eNetMF算法.基于5个实际的网络数据集,我们评估了eNetMF学习到的网络低维表示在多标签节点分类和边预测上的有效性.实验结果表明,使用eNetMF替代NetMF后在后续得到的多标签分类性能指标上几乎没有损失,但在处理大规模数据时有超过40倍的加速与内存用量节省.在一台32核的机器上,eNetMF仅需约1.3 h即可对含一百多万节点的YouTube数据学习到网络嵌入,内存用量仅为120GB,并得到较高质量的分类结果.此外,最近被提出的网络嵌入算法NetSMF由于图稀疏化过程的内存需求太大,无法在256 GB内存的机器上处理两个较大的网络数据,而ProNE算法则在多标签分类的结果上表现不稳定,得到的Macro-F1值都比较差.因此,eNetMF算法在结果质量上明显优于NetSMF和ProNE算法.在边预测任务上,eNetMF算法也表现出与其它方法差不多甚至更好的性能."""

# calculate the probability of `machine learning`, `artificial intelligence`, `language model` for GPT-3 paper
print("=== Span Probability ===")
for span in ["机器学习", "网络嵌入", "随机化特征值分解"]:
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
for i in range(4):
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
