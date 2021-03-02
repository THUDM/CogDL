Models of CogDL
===============

CogDL now supports the following models for different tasks:

- unsupervised node classification (无监督结点分类): ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512), DNGR [(Cao et al, AAAI'16)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12423/11715).

- semi-supervised node classification (半监督结点分类): SGC-PN [(Zhao & Akoglu, 2019)](https://arxiv.org/abs/1909.12223), Graph U-Net [(Gao et al., 2019)](https://arxiv.org/abs/1905.05178), MixHop [(Abu-El-Haija et al., ICML'19)](https://arxiv.org/abs/1905.00067), DR-GAT [(Zou et al., 2019)](https://arxiv.org/abs/1907.02237), GAT [(Veličković et al., ICLR'18)](https://arxiv.org/abs/1710.10903), DGI [(Veličković et al., ICLR'19)](https://arxiv.org/abs/1809.10341), GCN [(Kipf et al., ICLR'17)](https://arxiv.org/abs/1609.02907), GraphSAGE [(Hamilton et al., NeurIPS'17)](https://arxiv.org/abs/1706.02216), Chebyshev [(Defferrard et al., NeurIPS'16)](https://arxiv.org/abs/1606.09375).

- heterogeneous node classification (异构结点分类): GTN [(Yun et al, NeurIPS'19)](https://arxiv.org/abs/1911.06455), HAN [(Xiao et al, WWW'19)](https://arxiv.org/abs/1903.07293), PTE [(Tang et al, KDD'15)](https://arxiv.org/abs/1508.00200), Metapath2vec [(Dong et al, KDD'17)](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf), Hin2vec [(Fu et al, CIKM'17)](https://dl.acm.org/doi/10.1145/3132847.3132953).

- link prediction (链接预测): ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), NetSMF [(Qiu et at, WWW'19)](https://arxiv.org/abs/1906.11156), SDNE [(Wang et al, KDD'16)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf).

- multiplex link prediction (多重边链接预测): GATNE [(Cen et al, KDD'19)](https://arxiv.org/abs/1905.01669), NetMF [(Qiu et al, WSDM'18)](http://arxiv.org/abs/1710.02971), ProNE [(Zhang et al, IJCAI'19)](https://www.ijcai.org/Proceedings/2019/0594.pdf), Node2vec [(Grover et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939754), DeepWalk [(Perozzi et al, KDD'14)](http://arxiv.org/abs/1403.6652), LINE [(Tang et al, WWW'15)](http://arxiv.org/abs/1503.03578), Hope [(Ou et al, KDD'16)](http://dl.acm.org/citation.cfm?doid=2939672.2939751), GraRep [(Cao et al, CIKM'15)](http://dl.acm.org/citation.cfm?doid=2806416.2806512).

- unsupervised graph classification (无监督图分类): Infograph [(Sun et al, ICLR'20)](https://openreview.net/forum?id=r1lfF2NYvH), Graph2Vec [(Narayanan et al, CoRR'17)](https://arxiv.org/abs/1707.05005), DGK [(Yanardag et al, KDD'15)](https://dl.acm.org/doi/10.1145/2783258.2783417).

- supervised graph classification (有监督图分类): GIN [(Xu et al, ICLR'19)](https://openreview.net/forum?id=ryGs6iA5Km), DiffPool [(Ying et al, NeuIPS'18)](https://arxiv.org/abs/1806.08804), SortPool [(Zhang et al, AAAI'18)](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf), PATCHY_SAN [(Niepert et al, ICML'16)](https://arxiv.org/pdf/1605.05273.pdf), DGCNN [(Wang et al, ACM Transactions on Graphics'17)](https://arxiv.org/abs/1801.07829).

> `metis` is required to run ClusterGCN, you can follow the following steps to install `metis`.
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
6) pip install metis

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

for SGC-PN:

- --dropout, Dropout rate in SGC-PN. Default is 0.01;
- --num-layers, Number of layers in SGC-PN. Default is 40;
- --norm-mode, Mode for PairNorm in SGC-PN. Default is "PN";
- --norm-scale, Row-normalization scale in SGC-PN. Default is 10;

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
[1] Zhao, Lingxiao, and Leman Akoglu. "Pairnorm: Tackling oversmoothing in gnns." arXiv preprint arXiv:1909.12223 (2019).

[2] Sun, Fan-Yun, Jordan Hoffmann, and Jian Tang. "InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization." arXiv preprint arXiv:1908.01000 (2019).

[3] Qiu, Jiezhong, Yuxiao Dong, Hao Ma, Jian Li, Chi Wang, Kuansan Wang, and Jie Tang. "Netsmf: Large-scale network embedding as sparse matrix factorization." In The World Wide Web Conference, pp. 1509-1520. 2019.

[4] Cen, Yukuo, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, and Jie Tang. "Representation learning for attributed multiplex heterogeneous network." In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1358-1368. 2019.

[5] Zhang, Jie, Yuxiao Dong, Yan Wang, Jie Tang, and Ming Ding. "ProNE: fast and scalable network representation learning." In Proc. 28th Int. Joint Conf. Artif. Intell., IJCAI, pp. 4278-4284. 2019.

[6] Zou, Xu, Qiuye Jia, Jianwei Zhang, Chang Zhou, Hongxia Yang, and Jie Tang. "Dimensional Reweighting Graph Convolutional Networks." arXiv preprint arXiv:1907.02237 (2019).

[7] Gao, Hongyang, and Shuiwang Ji. "Graph u-nets." arXiv preprint arXiv:1905.05178 (2019).

[8] Abu-El-Haija, Sami, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. "Mixhop: Higher-order graph convolutional architectures via sparsified neighborhood mixing." arXiv preprint arXiv:1905.00067 (2019).

[9] Veličković, Petar, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R. Devon Hjelm. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).

[10] Yun, Seongjun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J. Kim. "Graph Transformer Networks." In Advances in Neural Information Processing Systems, pp. 11960-11970. 2019.

[11] Wang, Xiao, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S. Yu. "Heterogeneous graph attention network." In The World Wide Web Conference, pp. 2022-2032. 2019.

[12] Xu, Keyulu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).

[13] Qiu, Jiezhong, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. "Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec." In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 459-467. 2018.

[14] Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).

[15] Ying, Zhitao, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. "Hierarchical graph representation learning with differentiable pooling." In Advances in neural information processing systems, pp. 4800-4810. 2018.

[16] Zhang, Muhan, Zhicheng Cui, Marion Neumann, and Yixin Chen. "An end-to-end deep learning architecture for graph classification." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

[17] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[18] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." In Advances in neural information processing systems, pp. 1024-1034. 2017.

[19] Dong, Yuxiao, Nitesh V. Chawla, and Ananthram Swami. "metapath2vec: Scalable representation learning for heterogeneous networks." In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 135-144. 2017.

[20] Fu, Tao-yang, Wang-Chien Lee, and Zhen Lei. "Hin2vec: Explore meta-paths in heterogeneous information networks for representation learning." In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, pp. 1797-1806. 2017.

[21] Narayanan, Annamalai, Mahinthan Chandramohan, Rajasekar Venkatesan, Lihui Chen, Yang Liu, and Shantanu Jaiswal. "graph2vec: Learning distributed representations of graphs." arXiv preprint arXiv:1707.05005 (2017).

[22] Wang, Yue, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. "Dynamic graph cnn for learning on point clouds." ACM Transactions on Graphics (TOG) 38, no. 5 (2019): 1-12.

[23] Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 855-864. 2016.

[24] Ou, Mingdong, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. "Asymmetric transitivity preserving graph embedding." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1105-1114. 2016.

[25] Wang, Daixin, Peng Cui, and Wenwu Zhu. "Structural deep network embedding." In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1225-1234. 2016.

[26] Cao, Shaosheng, Wei Lu, and Qiongkai Xu. "Deep neural networks for learning graph representations." In Thirtieth AAAI conference on artificial intelligence. 2016.

[27] Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." In Advances in neural information processing systems, pp. 3844-3852. 2016.

[28] Niepert, Mathias, Mohamed Ahmed, and Konstantin Kutzkov. "Learning convolutional neural networks for graphs." In International conference on machine learning, pp. 2014-2023. 2016.

[29] Tang, Jian, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. "Line: Large-scale information network embedding." In Proceedings of the 24th international conference on world wide web, pp. 1067-1077. 2015.

[30] Cao, Shaosheng, Wei Lu, and Qiongkai Xu. "Grarep: Learning graph representations with global structural information." In Proceedings of the 24th ACM international on conference on information and knowledge management, pp. 891-900. 2015.

[31] Tang, Jian, Meng Qu, and Qiaozhu Mei. "Pte: Predictive text embedding through large-scale heterogeneous text networks." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1165-1174. 2015.

[32] Yanardag, Pinar, and S. V. N. Vishwanathan. "Deep graph kernels." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1365-1374. 2015.

[33] Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701-710. 2014.

