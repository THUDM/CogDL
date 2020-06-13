Models of CogDL
===============

CogDL now supports the following models for different tasks:

- unsupervised node classification: ProNE [(Zhang et al, IJCAI'19)](http://arxiv.org/abs/1806.02623), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), NetSMF [(Qiu et at, WWW'19)](http://arxiv.org/abs/1710.02971), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512), DNGR [(Cao et al, AAAI'16)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715).

- semi-supervised node classification: NSGCN [(Zhang et al., 2019)](https://arxiv.org/), DR-GAT [(Zou et al., 2019)](https://arxiv.org/), DR-GCN [(Zou et al., 2019)](https://arxiv.org/), GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903), GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907), FastGCN [(Chen, Ma et al., ICLR'18)](https://arxiv.org/abs/1801.10247), GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216).

- heterogeneous node classification: GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455), HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293), PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200), Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf), Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953).

- link prediction: ProNE [(Zhang et al, IJCAI'19)](http://arxiv.org/abs/1806.02623), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), NetSMF [(Qiu et at, WWW'19)](http://arxiv.org/abs/1710.02971), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512).

- multiplex link prediction: GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), ProNE [(Zhang et al, IJCAI'19)](http://arxiv.org/abs/1806.02623), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512).

- unsupervised graph classification: Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH), Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005), DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417).

- supervised graph classification: GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km), DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804), SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf), PATCH_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf), DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829).

## Specific parameters

for DeepWalk and node2vec:

- --walk-num, the number of random walks to start at each node; the default is 10;
- --walk-length, Length of walk start at each node. Default is 50;
- --worker, Number of parallel workers. Default is 10;
- --window-size, Window size of skip-gram model. Default is 10;
- --iteration, Number of iterations. Default is 10;
- --q, Parameter in node2vec. Default is 1.0;
- --p, Parameter in node2vec. Default is 1.0;

for LINE:

- --order, Order of proximity in LINE. Default is 3 for 1+2;
- --alpha, Initial earning rate of SGD. Default is 0.025;
- --batch-size, Batch size in SGD training process. Default is 100;
- --negative, Number of negative nodes in sampling. Default is 5;

for HOPE:

- --beta, Parameter of katz for HOPE. Default is 0.01;

for Grarep:

- --step, Number of matrix step in GraRep and ProNE. Default is 5;

for NetMF:

- --window-size, Window size of deepwalk matrix. Default is 10;
- --is-large, Large or small for NetMF;
- --negative, Number of negative nodes in sampling. Default is 5;
- --rank, Number of Eigenpairs in NetMF, default is 256;

for NetSMF:

- --window-size, Window size of approximate matrix. Default is 10;
- --negative, Number of negative nodes in sampling. Default is 5;
- --round, Number of round in NetSMF. Default is 100;
- --worker, Number of parallel workers. Default is 10;

for ProNE:

- --step, Number of items in the chebyshev expansion. Default is 5;
- --theta, Parameter of ProNE. Default is 0.5;
- --mu, Parameter of ProNE. Default is 0.2;

for GCN and DR-GCN:

- --hidden-size, The size of hidden layer. Default=16;
- --num-layers, The number of GCN layer. Default=2;
- --dropout, The dropout probability. Default=0.5;

for GAT and DR-GAT:

- --hidden-size, The size of hidden layer. Default=8;
- --num-heads, The number of heads in attention mechanism. Default=8;
- --dropout, The dropout probability. Default=0.6;

for Graphsage:

- --hidden-size, The size of hidden layer. Default=8;
- --num-layers, The number of Graphsage. Default=2;
- --sample-size, The List of number of neighbor samples for each node in Graphsage. Default=10, 10;
- --dropout, The dropout probability. Default=0.5;


## References
[1] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In KDD, pages 701–710, 2014.

[2] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. Line: Large-scale information network embedding. In WWW, pages 1067–1077, 2015.

[3] Aditya Grover and Jure Leskovec.  node2vec: Scalable feature learning for networks. In KDD, pages 855–864, 2016.

[4] Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. Asymmetric transitivity preserving graph embedding. In KDD, pages 1105–1114, 2016.

[5] Shaosheng Cao, Wei Lu, and Qiongkai Xu.  Grarep: Learning graph representations with global structural information. In CIKM, pages 891–900, 2015.

[6] Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec. In WSDM, pages 459–467, 2018.

[7] Qiu, J., et al. NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorizationl[C]. in The World Wide Web Conference. ACM,  2019: p. 1509-1520.

[8] Zhang, J., et al., ProNE: Fast and Scalable Network Representation Learning. In IJCAI, 2019.

[9] Kipf, T.N. and M. Welling, Semi-Supervised Classification with Graph Convolutional Networks[J]. In ICLR, 2016.

[10] Veličković, P., et al., Graph Attention Networks[J]. In ICLR, 2017.

[11] Hamilton, W.L., R. Ying, and J. Leskovec, Inductive Representation Learning on Large Graphs[J]. In NIPS 2017.

[12] Zhang et al.  Revisiting Graph Convolutional Networks: Neighborhood Aggregation and Network Sampling.

[13] Zou X, Jia Q, Zhang J, et al. Dimensional Reweighting Graph Convolutional Networks[J]. arXiv preprint arXiv:1907.02237, 2019.
