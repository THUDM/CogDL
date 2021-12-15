import os
from cogdl.oag import oagbert
import torch
import torch.nn.functional as F
import numpy as np


# load time
tokenizer, model = oagbert("oagbert-v2-zh-sim")
model.eval()

# Paper 1
title = "国内外尾矿坝事故致灾因素分析"
abstract = "通过搜集已有尾矿坝事故资料,分析了国内外尾矿坝事故与坝高、筑坝工艺及致灾因素的关系。对147起尾矿坝事故的分析研究表明, 引起尾矿坝事故的主要因素为降雨,其次为地震、管理等;"

# Encode first paper
(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(title=title, abstract=abstract)
_, paper_embed_1 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# Positive Paper 2
title = "尾矿库工程特性及其安全监控系统研究"
abstract = "总结了尾矿坝工程的特殊性和复杂性.为了保证尾矿坝在全生命周期(包括运行期及其闭库后)的安全,发展尾矿库安全监控系统具有重要意义.提出了尾矿库安全监控的基础框架,分析了尾矿库安全监测的主要内容及关键问题,为保证尾矿库的安全提供强有力的科学和技术依据."
# Encode second paper
(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(title=title, abstract=abstract)
_, paper_embed_2 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# Negative Paper 3
title = "Windows下EPA技术的研究与改进"
abstract = "该文对Windows下rookit的几种检测技术进行了比较和研究,并着重分析了基于可执行路径分析(EPA)技术.同时还讨论了其在Win2k下的代码实现,并提出改进方案。"
# encode third paper
(
    input_ids,
    input_masks,
    token_type_ids,
    masked_lm_labels,
    position_ids,
    position_ids_second,
    masked_positions,
    num_spans,
) = model.build_inputs(title=title, abstract=abstract)
_, paper_embed_3 = model.bert.forward(
    input_ids=torch.LongTensor(input_ids).unsqueeze(0),
    token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
    attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
    output_all_encoded_layers=False,
    checkpoint_activations=False,
    position_ids=torch.LongTensor(position_ids).unsqueeze(0),
    position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
)

# calulate text similarity

# normalize
paper_embed_1 = F.normalize(paper_embed_1, p=2, dim=1)
paper_embed_2 = F.normalize(paper_embed_2, p=2, dim=1)
paper_embed_3 = F.normalize(paper_embed_3, p=2, dim=1)

# cosine sim.
sim12 = torch.mm(paper_embed_1, paper_embed_2.transpose(0, 1))
sim13 = torch.mm(paper_embed_1, paper_embed_3.transpose(0, 1))
print(sim12, sim13)
