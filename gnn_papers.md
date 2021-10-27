# GNN paper lists

We select the 100 most influential GNN papers and 100 recent SOTA GNN papers with 10 different topics.

## [Content](#content)

<table>
<tr><td><a href="#gnn-architecture">1. GNN Architecture</a></td></tr> 
<tr><td><a href="#large-scale-training">2. Large-scale Training</a></td></tr>
<tr><td><a href="#self-supervised-learning-and-pre-training">3. Self-supervised Learning and Pre-training</a></td></tr>
<tr><td><a href="#oversmoothing-and-deep-gnns">4. Oversmoothing and Deep GNNs</a></td></tr>
<tr><td><a href="#graph-robustness">5. Graph Robustness</a></td></tr>
<tr><td><a href="#explainability">6. Explainability</a></td></tr>
<tr><td><a href="#expressiveness-and-generalisability">7. Expressiveness and Generalisability</a></td></tr>
<tr><td><a href="#heterogeneous-gnns">8. Heterogeneous GNNs</a></td></tr>
<tr><td><a href="#gnns-for-recommendation">9. GNNs for Recommendation</a></td></tr>
<tr><td><a href="#chemistry-and-biology">10. Chemistry and Biology</a></td></tr>
</table>

## [GNN Architecture](#content)

### Most Influential

1. **Semi-Supervised Classification with Graph Convolutional**. Thomas N. Kipf, Max Welling. NeuIPS'17. [paper](https://arxiv.org/abs/1609.02907)
2. **Graph Attention Networks**. Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. ICLR'18. [paper](https://arxiv.org/abs/1710.10903)
3. **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**. NeuIPS'16. [paper](https://arxiv.org/abs/1606.09375)
4. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank**. Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann. ICLR'19. [paper](https://arxiv.org/abs/1810.05997) 
5. **Gated Graph Sequence Neural Networks**. Li, Yujia N and Tarlow, Daniel and Brockschmidt, Marc and Zemel, Richard. ICLR'16. [paper](https://arxiv.org/abs/1511.05493)
6. **Inductive Representation Learning on Large Graphs**. William L. Hamilton, Rex Ying, Jure Leskovec. NeuIPS'17. [paper](https://arxiv.org/abs/1706.02216)
7. **Deep Graph Infomax**. Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm. ICLR'19. [paper](https://openreview.net/pdf?id=rklz9iAcKQ)
8. **Representation Learning on Graphs with Jumping Knowledge Networks**. Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka. ICML'18. [paper](https://arxiv.org/abs/1806.03536)
9. **DeepGCNs: Can GCNs Go as Deep as CNNs?**. Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem. ICCV'19. [paper](https://arxiv.org/abs/1904.03751)
10. **DropEdge: Towards Deep Graph Convolutional Networks on Node Classificatio**. Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang. ICLR'20. [paper](https://arxiv.org/abs/1907.10903)

### Recent SOTA

1. **Training Graph Neural Networks with 1000 Layers**. Guohao Li, Matthias Müller, Bernard Ghanem, Vladlen Koltun. ICML'21. [paper](https://arxiv.org/abs/2106.07476) 
2. **Graph Random Neural Network for Semi-Supervised Learning on Graphs**. Wenzheng Feng, Jie Zhang, Yuxiao Dong, Yu Han, Huanbo Luan, Qian Xu, Qiang Yang, Evgeny Kharlamov, Jie Tang. NeuIPS'20. [paper](https://arxiv.org/abs/2005.11079)
3. **Simple and Deep Graph Convolutional Networks**. Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, Yaliang Li. ICML'20. [paper](https://arxiv.org/abs/2007.02133)
4. **Combining Label Propagation and Simple Models Out-performs Graph Neural Networks**. Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, Austin R. Benson. ICLR'21. [paper](https://arxiv.org/abs/2010.13993)
5. **Graph Attention Multi-Layer Perceptron**. Wentao Zhang, Ziqi Yin, Zeang Sheng, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui. [paper](https://arxiv.org/abs/2108.10097)
6. **How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision**. Dongkwan Kim, Alice Oh. ICLR'21. [paper](https://openreview.net/forum?id=Wi5KUNlqWty)
7. **Towards Deeper Graph Neural Networks**. Meng Liu, Hongyang Gao, Shuiwang Ji. KDD'20. [paper](https://arxiv.org/abs/2007.09296)
8. **Graph Traversal with Tensor Functionals: A Meta-Algorithm for Scalable Learning**. Elan Markowitz, Keshav Balasubramanian, Mehrnoosh Mirtaheri, Sami Abu-El-Haija, Bryan Perozzi, Greg Ver Steeg, Aram Galstyan. ICLR'21. [paper](https://arxiv.org/abs/2102.04350)
9. **MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing**. Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, Aram Galstyan. ICML'19. [paper](https://arxiv.org/abs/1905.00067)
10. **Diffusion Improves Graph Learning**. Johannes Klicpera, Stefan Weißenberger, Stephan Günnemann. NeuIPS'19. [paper](https://arxiv.org/abs/1911.05485)


## [Large-scale Training](#content)

### Most Influential

1. **Inductive Representation Learning on Large Graphs**. William L. Hamilton, Rex Ying, Jure Leskovec. NeuIPS'17. [paper](https://arxiv.org/abs/1706.02216)
2. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling**. Jie Chen, Tengfei Ma, Cao Xiao. ICLR'18. [paper](https://arxiv.org/abs/1801.10247)
3. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**. Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh. KDD'19. [paper](https://arxiv.org/abs/1905.07953)
4. **GraphSAINT: Graph Sampling Based Inductive Learning Method**. Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna. ICLR'20. [paper](https://arxiv.org/abs/1907.04931)
5. **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings**. Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec. ICML'21. [paper](https://arxiv.org/abs/2106.05609)
6. **Scaling Graph Neural Networks with Approximate PageRank**. Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann. KDD'20. [paper](https://arxiv.org/abs/2007.01570)
7. **Stochastic training of graph convolutional networks with variance reduction**. *Jianfei Chen, Jun Zhu, and Le Song.* ICML'18. [paper](https://arxiv.org/abs/1710.10568)
8. **Adaptive sampling towards fast graph representation learning**. Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. NeuIPS'18. [paper](https://papers.nips.cc/paper/2018/file/01eee509ee2f68dc6014898c309e86bf-Paper.pdf)
9. **SIGN: Scalable Inception Graph Neural Networks**. Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, Federico Monti. [paper](https://arxiv.org/abs/2004.11198)
10. **Simplifying Graph Convolutional Networks**. Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger. ICML'19. [paper](https://arxiv.org/abs/1902.07153)

### Recent SOTA

1. **GraphSAINT: Graph Sampling Based Inductive Learning Method**. Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna. ICLR'20. [paper](https://arxiv.org/abs/1907.04931)
2. **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings**. Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec. ICML'21. [paper](https://arxiv.org/abs/2106.05609)
3. **Deep Graph Neural Networks with Shallow Subgraph Samplers**. Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor Prasanna, Long Jin, Ren Chen. [paper](https://arxiv.org/abs/2012.01380)
4. **Scalable Graph Neural Networks via Bidirectional Propagation**. Ming Chen, Zhewei Wei, Bolin Ding, Yaliang Li, Ye Yuan, Xiaoyong Du, Ji-Rong Wen. NeuIPS'20. [paper](https://arxiv.org/abs/2010.15421)
5. **A Unified Lottery Ticket Hypothesis for Graph Neural Networks**. Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, Zhangyang Wang. ICML'21. [paper](https://arxiv.org/abs/2102.06790).
6. **Scaling Graph Neural Networks with Approximate PageRank**. Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann. KDD'20. [paper](https://arxiv.org/abs/2007.01570)
7. **Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training**. Chuxiong Sun, Hongming Gu, Jie Hu. [paper](https://arxiv.org/abs/2104.09376)
8. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**. Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh. KDD'19. [paper](https://arxiv.org/abs/1905.07953)
9. **GraphZoom: A Multi-level Spectral Approach for Accurate and Scalable Graph Embedding**. Chenhui Deng, Zhiqiang Zhao, Yongyu Wang, Zhiru Zhang, Zhuo Feng. ICLR'20. [paper](https://arxiv.org/abs/1910.02370)
10. **Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs**. 	Jialin Dong, Da Zheng, Lin F. Yang, Geroge Karypis. KDD'21. [paper](https://arxiv.org/abs/2106.06150)


## [Self-supervised Learning and Pre-training](#content)

### Most Influential

1. **Strategies for pre-training graph neural networks.** *Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande,  Leskovec Jure.* ICLR 2020. [paper](https://openreview.net/forum?id=HJlWWJSFDH)
2. **Deep graph infomax.** *Velikovi Petar, Fedus William, Hamilton William L, Li Pietro, Bengio Yoshua, Hjelm R Devon.* ICLR 2019. [paper](https://arxiv.org/abs/1809.10341)
3. **Inductive representation learning on large graphs.** *Hamilton Will, Zhitao Ying, Leskovec Jure.* NeurIPS 2017. [paper](https://arxiv.org/abs/1706.02216)
4. **Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization.** *Sun Fan-Yun, Hoffmann Jordan, Verma Vikas, Tang Jian.* ICLR 2020. [paper](https://arxiv.org/pdf/1908.01000.pdf)
5. **GCC: Graph contrastive coding for graph neural network pre-training.** *Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, Jie Tang.* KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168)
6. **Contrastive multi-view representation learning on graphs.** *Hassani Kaveh, Khasahmadi Amir Hosein.* ICML 2020. [paper](https://arxiv.org/abs/2006.05582)
7. **Graph contrastive learning with augmentations.** *Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen.* NeurIPS 2020. [paper](https://arxiv.org/abs/2010.13902)
8. **GPT-GNN: Generative pre-training of graph neural networks.** *Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, Yizhou Sun.* KDD 2020. [paper](https://arxiv.org/abs/2006.15437)
9. **When does self-supervision help graph convolutional networks?.** *Yuning You, Tianlong Chen, Zhangyang Wang, Yang Shen.* ICML 2020. [paper](https://arxiv.org/abs/2006.09136)
10. **Deep graph contrastive representation learning.** *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang.* GRL+@ICML 2020. [paper](https://arxiv.org/abs/2006.04131)

### Recent SOTA

1. **Graph Contrastive Learning Automated.** *Yuning You, Tianlong Chen, Yang Shen, Zhangyang Wang.* ICML 2021. [paper](https://arxiv.org/abs/2106.07594)
2. **Graph contrastive learning with adaptive augmentation.** *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang.* WWW 2021. [paper](https://arxiv.org/abs/2010.14945)
3. **Self-supervised Graph-level Representation Learning with Local and Global Structure.** *Minghao Xu, Hang Wang, Bingbing Ni, Hongyu Guo, Jian Tang.* ICML 2021. [paper](https://arxiv.org/pdf/2106.04113)
4. **Negative Sampling Strategies for Contrastive Self-Supervised Learning of Graph Representations.** *Hakim Hafidi, Mounir Ghogho, Philippe Ciblat, Ananthram Swami.* Signal Processing 2021. [paper](https://www.sciencedirect.com/science/article/pii/S0165168421003479)
5. **Learning to pre-train graph neural networks.** *Yuanfu Lu, Xunqiang Jiang, Yuan Fang, Chuan Shi.* AAAI 2021. [paper](http://shichuan.org/doc/101.pdf)
6. **Graph representation learning via graphical mutual information maximization.** *Zhen Peng, Wenbing Huang, Minnan Luo, Qinghua Zheng, Yu Rong, Tingyang Xu, Junzhou Huang.* WWW 2020. [paper](https://arxiv.org/abs/2002.01169)
7. **Structure-Aware Hard Negative Mining for Heterogeneous Graph Contrastive Learning.** *Yanqiao Zhu, Yichen Xu, Hejie Cui, Carl Yang, Qiang Liu, Shu Wu.* arXiv preprint arXiv:2108.13886 2021. [paper](https://arxiv.org/abs/2108.13886)
8. **Self-supervised graph representation learning via global context prediction.** *Zhen Peng, Yixiang Dong, Minnan Luo, Xiao-Ming Wu, Qinghua Zheng.* arXiv preprint arXiv:2003.01604 2020. [paper](https://arxiv.org/abs/2003.01604)
9. **CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction.** *Chengshuai Zhao, Shuai Liu, Feng Huang, Shichao Liu, Wen Zhang.* IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0517.pdf)
10. **Pairwise Half-graph Discrimination: A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks.** *Pengyong Li, Jun Wang, Ziliang Li, Yixuan Qiao, Xianggen Liu, Fei Ma, Peng Gao, Sen Song, Guotong Xie.* IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0371.pdf)


## [Oversmoothing and Deep GNNs](#content)

### Most Influential

1. **Representation Learning on Graphs with Jumping Knowledge Networks.** *Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka.* ICML 2018. [paper](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf)
2. **Deeper insights into graph convolutional networks for semi-supervised learning.** *Qimai Li, Zhichao Han, Xiao-ming Wu.* AAAI 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16098)
3. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank.** *Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann.* ICLR 2019. [paper](https://openreview.net/forum?id=H1gL-2A9Ym)
4. **DeepGCNs: Can GCNs Go as Deep as CNNs?** *Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem.* ICCV 2019. [paper](https://arxiv.org/abs/1904.03751)
5. **Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks.** *Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, Quanquan Gu.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-6006)
6. **DeeperGCN: All You Need to Train Deeper GCNs.** *Guohao Li, Chenxin Xiong, Ali Thabet, Bernard Ghanem.* arXiv 2020. [paper](https://arxiv.org/abs/2006.07739)
7. **PairNorm: Tackling Oversmoothing in GNNs.** *Lingxiao Zhao, Leman Akoglu.* ICLR 2020. [paper](https://openreview.net/forum?id=rkecl1rtwB)
8. **DropEdge: Towards Deep Graph Convolutional Networks on Node Classification.** *Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang.* ICLR 2020. [paper](https://openreview.net/pdf?id=Hkx1qkrKPr)
9. **Simple and Deep Graph Convolutional Networks.** *Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, Yaliang Li.* ICML 2020. [paper](http://proceedings.mlr.press/v119/chen20v.html)
10. **Towards Deeper Graph Neural Networks.** *Meng Liu, Hongyang Gao, and Shuiwang Ji.* KDD 2020. [paper](https://dl.acm.org/doi/10.1145/3394486.3403076)

### Recent SOTA

1. **Towards Deeper Graph Neural Networks with Differentiable Group Normalization.** *Kaixiong Zhou, Xiao Huang, Yuening Li, Daochen Zha, Rui Chen, Xia Hu.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/33dd6dba1d56e826aac1cbf23cdcca87-Abstract.html)
2. **Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks.** *Yimeng Min, Frederik Wenkel, Guy Wolf.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/a6b964c0bb675116a15ef1325b01ff45-Abstract.html)
3. **Optimization and Generalization Analysis of Transduction through Gradient Boosting and Application to Multi-scale Graph Neural Networks.** *Kenta Oono, Taiji Suzuki.* NeurIPS 2020. [paper](https://papers.nips.cc//paper/2020/hash/dab49080d80c724aad5ebf158d63df41-Abstract.html)
4. **On the Bottleneck of Graph Neural Networks and its Practical Implications.** *Uri Alon, Eran Yahav.* ICLR 2021. [paper](https://openreview.net/forum?id=i80OPhOCVH2)
5. **Simple Spectral Graph Convolution.** *Hao Zhu, Piotr Koniusz.* ICLR 2021. [paper](https://openreview.net/forum?id=CYO5T-YjWZV)
6. **Training Graph Neural Networks with 1000 Layers.** *Guohao Li, Matthias Müller, Bernard Ghanem, Vladlen Koltun.* ICML 2021. [paper](http://proceedings.mlr.press/v139/li21o.html)
7. **Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth.** *Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, Kenji Kawaguchi.* ICML 2021. [paper](http://proceedings.mlr.press/v139/xu21k.html)
8. **GRAND: Graph Neural Diffusion.** *Ben Chamberlain, James Rowbottom, Maria I Gorinova, Michael Bronstein, Stefan Webb, Emanuele Rossi.* ICML 2021. [paper](http://proceedings.mlr.press/v139/chamberlain21a.html)
9. **Directional Graph Networks.** *Dominique Beani, Saro Passaro, Vincent Létourneau, Will Hamilton, Gabriele Corso, Pietro Lió.* ICML 2021. [paper](http://proceedings.mlr.press/v139/beani21a.html)
10. **Improving Breadth-Wise Backpropagation in Graph Neural Networks Helps Learning Long-Range Dependencies.** *Denis Lukovnikov, Asja Fischer.* ICML 2021. [paper](http://proceedings.mlr.press/v139/lukovnikov21a.html)


## [Graph Robustness](#content)

### Most Influential

1. **Adversarial attacks on neural networks for graph data**. *Zügner Daniel, Akbarnejad Amir, Günnemann Stephan*. KDD 2018. [paper](https://arxiv.org/abs/1805.07984) [code](https://github.com/danielzuegner/nettack)
2. **Adversarial attack on graph structured data**. *Dai Hanjun, Li Hui, Tian Tian, Huang Xin, Wang Lin, Zhu Jun, Song Le*. ICML 2018. [paper](https://arxiv.org/abs/1806.02371) [code](https://github.com/Hanjun-Dai/graph_adversarial_attack)
3. **Adversarial attacks on graph neural networks via meta learning**. *Zügner Daniel, Günnemann Stephan*. ICLR 2019. [paper](https://arxiv.org/abs/1902.08412) [code](https://github.com/danielzuegner/gnn-meta-attack)
4. **Robust graph convolutional networks against adversarial attacks**. *Zhu Dingyuan, Zhang Ziwei, Cui Peng, Zhu Wenwu*. KDD 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330851?casa_token=cCf7RRIzp60AAAAA:CQRAqGd--EvKBZfUenQ0StGA9JwoyY0ZBVhZ0R6OF4n8Za3N0wopLpev6se5r6YT5BopGE8oj0fUfhc) [code](https://github.com/ZW-ZHANG/RobustGCN)
5. **Adversarial attacks on node embeddings via graph poisoning**. *Bojchevski Aleksandar, Günnemann Stephan*. ICML 2019. [paper](https://arxiv.org/abs/1809.01093) [code](https://github.com/abojchevski/node_embedding_attack)
6. **Topology attack and defense for graph neural networks: An optimization perspective**. *Xu Kaidi, Chen Hongge, Liu Sijia, Chen Pin-Yu, Weng Tsui-Wei, Hong Mingyi, Lin Xue*. IJCAI 2019. [paper](https://arxiv.org/abs/1906.04214)
7. **Adversarial examples on graph data: Deep insights into attack and defense**. *Wu Huijun, Wang Chen, Tyshetskiy Yuriy, Docherty Andrew, Lu Kai, Zhu Liming*. IJCAI 2019. [paper](https://arxiv.org/abs/1903.01610)
8. **Certifiable robustness and robust training for graph convolutional networks**. *Zügner Daniel, Günnemann Stephan*. KDD 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330905?casa_token=RKxopFEgP4MAAAAA:WPC4t9R_rTVRu2179sw3PlVLl5TzSDfYALMqw5cNSz53hiL4jSmT8_gQiQ8kn0N47GzWiJkXSo-FAzk)
9. **Graph adversarial training: Dynamically regularizing based on graph structure**. *Feng Fuli, He Xiangnan, Tang Jie, Chua Tat-Seng*. TKDE 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8924766/?casa_token=H51wrycvI6UAAAAA:-q7RBGcyWMUiSjJhonyV1zoIGDgVkVndSZ4E3B77XFmiGS31msJuoa-leKhW3aTuGV-XLfTi504)
10. **Adversarial attack and defense on graph data: A survey**. *Sun Lichao, Dou Yingtong, Yang Carl, Wang Ji, Yu Philip S, He Lifang, Li Bo*. arXiv preprint arXiv:1812.10528 2018. [paper](https://arxiv.org/abs/1812.10528)

### Recent SOTA

1. **TDGIA: Effective Injection Attacks on Graph Neural Networks**. *Zou Xu, Zheng Qinkai, Dong Yuxiao, Guan Xinyu, Kharlamov Evgeny, Lu Jialiang, Tang Jie*. KDD 2021. [paper](https://arxiv.org/abs/2106.06663) [code](https://github.com/THUDM/tdgia/)
2. **Gnnguard: Defending graph neural networks against adversarial attacks**. *Zhang Xiang, Zitnik Marinka*. NeurIPS 2020. [paper](https://arxiv.org/abs/2006.08149) [code](https://github.com/mims-harvard/GNNGuard)
3. **Attacking Graph Neural Networks at Scale**. *Geisler Simon, Zügner Daniel, Bojchevski Aleksandar, Günnemann Stephan*. AAAI workshop 2021. [paper](https://www.in.tum.de/fileadmin/w00bws/daml/attack-gnns-at-scale/Robust_GNNs_at_Scale__AAAI_DLG_Workshop_2021.pdf)
4. **Graph Random Neural Networks for Semi-Supervised Learning on Graphs**. *Feng Wenzheng, Zhang Jie, Dong Yuxiao, Han Yu, Luan Huanbo, Xu Qian, Yang Qiang, Kharlamov Evgeny, Tang Jie*. NeurIPS 2020. [paper](https://arxiv.org/abs/2005.11079) [code](https://github.com/THUDM/GRAND)
5. **Graph information bottleneck**. *Wu Tailin, Ren Hongyu, Li Pan, Leskovec Jure*. NeurIPS 2020. [paper](https://arxiv.org/abs/2010.12811) [code](https://github.com/snap-stanford/GIB)
6. **Information Obfuscation of Graph Neural Networks**. *Liao Peiyuan, Zhao Han, Xu Keyulu, Jaakkola Tommi, Gordon Geoffrey J, Jegelka Stefanie, Salakhutdinov Ruslan*. ICML 2021. [paper](http://proceedings.mlr.press/v139/liao21a.html) 
7. **Understanding Structural Vulnerability in Graph Convolutional Networks**. *Chen Liang, Li Jintang, Peng Qibiao, Liu Yang, Zheng Zibin, Yang Carl*. IJCAI 2021. [paper](https://arxiv.org/abs/2108.06280) [code](https://github.com/EdisonLeeeee/MedianGCN)
8. **Certified robustness of graph neural networks against adversarial structural perturbation**. *Wang Binghui, Jia Jinyuan, Cao Xiaoyu, Gong Neil Zhenqiang*. KDD 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467295?casa_token=mXWjPXMlT7QAAAAA:xp5Eja0yvHJBJ-PCBLIKQnkYmSkdggU-G7rH__G4ZjxccOuJ-jN5WkW-7X196tL0rtP9ZnKP0Sj8l-Y) [code](https://github.com/binghuiwang/CertifyGNN)
9. **Expressive 1-Lipschitz Neural Networks for Robust Multiple Graph Learning against Adversarial Attacks**. *Zhao Xin, Zhang Zeru, Zhang Zijie, Wu Lingfei, Jin Jiayin, Zhou Yang, Jin Ruoming, Dou Dejing, Yan Da*. ICML 2021. [paper](http://proceedings.mlr.press/v139/zhao21e.html)
10. **Reliable graph neural networks via robust aggregation**. *Geisler Simon, Zügner Daniel, Günnemann Stephan*. NeurIPS 2020. [paper](https://arxiv.org/abs/2010.15651) [code](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation)


## [Explainability](#content)

### Most Influential

1. **Explainability in graph neural networks: A taxonomic survey**. *Yuan Hao, Yu Haiyang, Gui Shurui, Ji Shuiwang*. ARXIV 2020. [paper](https://arxiv.org/pdf/2012.15445.pdf)
2. **Gnnexplainer: Generating explanations for graph neural networks**. *Ying Rex, Bourgeois Dylan, You Jiaxuan, Zitnik Marinka, Leskovec Jure*. NeurIPS 2019. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7138248/) [code](https://github.com/RexYing/gnn-model-explainer)
3. **Explainability methods for graph convolutional neural networks**. *Pope Phillip E, Kolouri Soheil, Rostami Mohammad, Martin Charles E, Hoffmann Heiko*. CVPR 2019.[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
4. **Parameterized Explainer for Graph Neural Network**. *Luo Dongsheng, Cheng Wei, Xu Dongkuan, Yu Wenchao, Zong Bo, Chen Haifeng, Zhang Xiang*. NeurIPS 2020. [paper](https://arxiv.org/abs/2011.04573) [code](https://github.com/flyingdoog/PGExplainer)
5. **Xgnn: Towards model-level explanations of graph neural networks**. *Yuan Hao, Tang Jiliang, Hu Xia, Ji Shuiwang*. KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403085). 
6. **Evaluating Attribution for Graph Neural Networks**. *Sanchez-Lengeling Benjamin, Wei Jennifer, Lee Brian, Reif Emily, Wang Peter, Qian Wesley, McCloskey Kevin, Colwell  Lucy, Wiltschko Alexander*. NeurIPS  2020.[paper](https://proceedings.neurips.cc/paper/2020/file/417fbbf2e9d5a28a855a11894b2e795a-Paper.pdf)
7. **PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks**. *Vu Minh, Thai My T.*. NeurIPS  2020.[paper](https://arxiv.org/pdf/2010.05788.pdf)
8. **Explanation-based Weakly-supervised Learning of Visual Relations with Graph Networks**. *Federico Baldassarre and Kevin Smith and Josephine Sullivan and Hossein Azizpour*. ECCV 2020.[paper](https://arxiv.org/pdf/2010.05788.pdf)
9. **GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media**. *Lu, Yi-Ju and Li, Cheng-Te*. ACL 2020.[paper](https://arxiv.org/pdf/2004.11648.pdf)
10. **On Explainability of Graph Neural Networks via Subgraph Explorations**. *Yuan Hao, Yu Haiyang, Wang Jie, Li Kang, Ji Shuiwang*. ICML 2021.[paper](https://arxiv.org/pdf/2102.05152.pdf)

### Recent SOTA

1. **Quantifying Explainers of Graph Neural Networks in Computational Pathology**. *Jaume Guillaume, Pati Pushpak, Bozorgtabar Behzad, Foncubierta Antonio, Anniciello Anna Maria, Feroce Florinda, Rau Tilman, Thiran Jean-Philippe, Gabrani Maria, Goksel Orcun*. Proceedings of the IEEECVF Conference on Computer Vision and Pattern Recognition CVPR 2021.[paper](https://arxiv.org/pdf/2011.12646.pdf)
2. **Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network**. *Wu Haoran, Chen Wei, Xu Shuang, Xu Bo*. NAACL 2021. [paper](https://aclanthology.org/2021.naacl-main.156.pdf)
3. **When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods**. *Faber Lukas, K. Moghaddam Amin, Wattenhofer Roger*. KDD 2021. [paper](https://dl.acm.org/doi/10.1145/3447548.3467283)
4. **Counterfactual Graphs for Explainable Classification of Brain Networks**. *Abrate Carlo, Bonchi Francesco*. Proceedings of the th ACM SIGKDD Conference on Knowledge Discovery  Data Mining KDD 2021. [paper](https://arxiv.org/pdf/2106.08640.pdf)
5. **Explainable Subgraph Reasoning for Forecasting on Temporal Knowledge Graphs**. *Zhen Han, Peng Chen, Yunpu Ma, Volker Tresp*. International Conference on Learning Representations ICLR 2021.[paper](https://iclr.cc/virtual/2021/poster/3378)
6. **Generative Causal Explanations for Graph Neural Networks**. *Lin Wanyu, Lan Hao, Li Baochun*. Proceedings of the th International Conference on Machine Learning ICML 2021.[paper](https://arxiv.org/pdf/2104.06643.pdf)
7. **Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity**. *Henderson Ryan, Clevert Djork-Arné, Montanari Floriane*. Proceedings of the th International Conference on Machine Learning ICML 2021.[paper](https://arxiv.org/pdf/2105.04854.pdf)
8. **Explainable Automated Graph Representation Learning with Hyperparameter Importance**. *Wang Xin, Fan Shuyi, Kuang Kun, Zhu Wenwu*. Proceedings of the th International Conference on Machine Learning ICML 2021.[paper](http://proceedings.mlr.press/v139/wang21f/wang21f.pdf)
9. **Higher-order explanations of graph neural networks via relevant walks**. *Schnake Thomas, Eberle Oliver, Lederer Jonas, Nakajima Shinichi, Schütt Kristof T, Müller Klaus-Robert, Montavon Grégoire*. arXiv preprint arXiv:2006.03589 2020. [paper](https://arxiv.org/pdf/2006.03589.pdf)
10. **HENIN: Learning Heterogeneous Neural Interaction Networks for Explainable Cyberbullying Detection on Social Media**. *Chen, Hsin-Yu and Li, Cheng-Te*. EMNLP 2020. [paper](https://www.aclweb.org/anthology/2020.emnlp-main.200/)


## [Expressiveness and Generalisability](#content)

### Most Influential

1. **How Powerful are Graph Neural Networks?** *Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka.* ICLR 2019. [paper](https://arxiv.org/abs/1810.00826)
2. **Invariant and Equivariant Graph Networks.** *Haggai Maron, Heli Ben-Hamu, Nadav Shamir, Yaron Lipman.* ICLR 2019. [paper](https://openreview.net/forum?id=Syx72jC9tm)
3. **Understanding Attention and Generalization in Graph Neural Networks.** *Boris Knyazev, Graham W. Taylor, Mohamed R. Amer.* NeurIPS 2019. [paper](https://arxiv.org/abs/1905.02850) 
4. **Provably Powerful Graph Networks.** *Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, Yaron Lipman.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-1275)
5. **Understanding the Representation Power of Graph Neural Networks in Learning Graph Topology.** *Nima Dehmamy, Albert-Laszlo Barabasi, Rose Yu.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-8876)
6. **On the equivalence between graph isomorphism testing and function approximation with GNNs.** *Zhengdao Chen, Soledad Villar, Lei Chen, Joan Bruna.* NeurIPS 2019. [paper](https://proceedings.neurips.cc/paper/2019/hash/71ee911dd06428a96c143a0b135041a4-Abstract.html)
7. **Universal Invariant and Equivariant Graph Neural Networks.** *Nicolas Keriven, Gabriel Peyré.* NeurIPS 2019. [paper](https://proceedings.neurips.cc/paper/2019/hash/ea9268cb43f55d1d12380fb6ea5bf572-Abstract.html)
8. **Stability and Generalization of Graph Convolutional Neural Networks.** *Saurabh Verma and Zhi-Li Zhang.* KDD 2019. [paper](https://dl.acm.org/doi/10.1145/3292500.3330956)
9. **Graph Neural Networks Exponentially Lose Expressive Power for Node Classification.** *Kenta Oono, Taiji Suzuki.* ICLR 2020. [paper](https://openreview.net/pdf?id=S1ldO2EFPr)
10. **Generalization and Representational Limits of Graph Neural Networks.** *Vikas Garg, Stefanie Jegelka, Tommi Jaakkola.* ICML 2020. [paper](https://www.mit.edu/~vgarg/GNN_CameraReady.pdf)

### Recent SOTA

1. **A PAC-Bayesian Approach to Generalization Bounds for Graph Neural Networks.** *Renjie Liao, Raquel Urtasun, Richard Zemel.* ICLR 2021. [paper](https://openreview.net/forum?id=TR-Nj6nFx42)
2. **Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective.** *Muhammet Balcilar, Guillaume Renton, Pierre Héroux, Benoit Gaüzère, Sébastien Adam, Paul Honeine.* ICLR 2021. [paper](https://openreview.net/forum?id=-qh0M9XWxnv)
3. **On Graph Neural Networks versus Graph-Augmented MLPs.** *Lei Chen, Zhengdao Chen, Joan Bruna.* ICLR 2021. [paper](https://openreview.net/forum?id=tiqI7w64JG2)
4. **Graph Convolution with Low-rank Learnable Local Filters.** *Xiuyuan Cheng, Zichen Miao, Qiang Qiu.* ICLR 2021. [paper](https://openreview.net/forum?id=9OHFhefeB86)
5. **From Local Structures to Size Generalization in Graph Neural Networks.** *Gilad Yehudai, Ethan Fetaya, Eli Meirom, Gal Chechik, Haggai Maron.* ICML 2021. [paper](https://arxiv.org/abs/2010.08853)
6. **A Collective Learning Framework to Boost GNN Expressiveness.** *Mengyue Hang, Jennifer Neville, Bruno Ribeiro.* ICML 2021. [paper](http://proceedings.mlr.press/v139/hang21a.html)
7. **Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks.** *Cristian Bodnar, Fabrizio Frasca, Yu Guang Wang, Nina Otter, Guido Montúfar, Pietro Liò, Michael Bronstein.* ICML 2021. [paper](https://arxiv.org/abs/2103.03212)
8. **Breaking the Limits of Message Passing Graph Neural Networks.** *Muhammet Balcilar, Pierre Heroux, Benoit Gauzere, Pascal Vasseur, Sebastien Adam, Paul Honeine.* *ICML 2021.* [paper](http://proceedings.mlr.press/v139/balcilar21a.html)
9. **Let's Agree to Degree: Comparing Graph Convolutional Networks in the Message-Passing Framework.** *Floris Geerts, Filip Mazowiecki, Guillermo A. Pérez.* ICML 2021. [paper](https://arxiv.org/abs/2004.02593)
10. **Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization.** *Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath.* ICML 2021. [paper](http://proceedings.mlr.press/v139/baranwal21a.html)


## [Heterogeneous GNNs](#content)

### Most Influential

1. **Heterogeneous Graph Attention Network**. *Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, P. Yu, Yanfang Ye* WWW 2019. [paper](https://arxiv.org/pdf/1903.07293.pdf) [code](https://github.com/taishan1994/pytorch_HAN)
2. **Representation Learning for Attributed Multiplex Heterogeneous Network**. *Yukuo Cen, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, Jie Tang* KDD 2019 [paper](https://arxiv.org/pdf/1905.01669.pdf) [code](https://github.com/THUDM/GATNE)
3. **ActiveHNE: Active Heterogeneous Network Embedding** *Xia Chen, Guoxian Yu, Jun Wang, Carlotta Domeniconi, Zhao Li, Xiangliang Zhang* IJCAI 2019 [paper](https://arxiv.org/pdf/1905.05659.pdf) 
4. **Hypergraph Neural Networks** *Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao*. AAAI 2019 [paper](https://arxiv.org/pdf/1809.09401.pdf) [code](https://github.com/iMoonLab/HGNN)
5. **Dynamic Hypergraph Neural Networks** *Jianwen Jiang, Yuxuan Wei, Yifan Feng, Jingxuan Cao, Yue Gao* IJCAI 2019 [paper](https://www.ijcai.org/proceedings/2019/0366.pdf) [code](https://github.com/iMoonLab/DHGNN)
6. **HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs.** *Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, Partha Talukdar* [paper](https://proceedings.neurips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) [code](https://github.com/malllabiisc/HyperGCN)
7. **Type-aware Anchor Link Prediction across Heterogeneous Networks based on Graph Attention Network.** *Xiaoxue Li, Yanmin Shang, Yanan Cao, Yangxi Li, Jianlong Tan, Yanbing Liu.* AAAI 2020 [paper](https://ojs.aaai.org/index.php/AAAI/article/download/5345/5201)
8. **Composition-based Multi-Relational Graph Convolutional Networks** *Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Partha Talukdar.* ICLR 2020 [paper](https://openreview.net/pdf?id=BylA_C4tPr) [code](https://github.com/malllabiisc/CompGCN)
9. **Hyper-SAGNN: a self-attention based graph neural network for hypergraphs.** *Ruochi Zhang, Yuesong Zou, Jian Ma.* ICLR 2020 [paper](https://openreview.net/pdf?id=ryeHuJBtPH) [code](https://github.com/ma-compbio/Hyper-SAGNN)
10. **Heterogeneous graph transformer** *Hu, Ziniu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun* WWW 2020 [paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3380027)[code](https://github.com/acbull/pyHGT)

### Recent SOTA

1. **An Attention-based Graph Neural Network for Heterogeneous Structural Learning** *Huiting Hong, Hantao Guo, Yu-Cheng Lin, Xiaoqing Yang, Zang Li, Jieping Ye*  AAAI 2020 [paper](http://shichuan.org/hin/topic/2020.An%20Attention-Based%20Graph%20Neural%20Network%20for%20Heterogeneous%20Structural%20Learning.pdf) [code](https://github.com/didi/hetsann)
2. **Heterogeneous Deep Graph Infomax** *Ren, Yuxiang, Bo Liu, Chao Huang, Peng Dai, Liefeng Bo, and Jiawei Zhang* AAAI 2020 workshop [paper](https://arxiv.org/abs/1911.08538) [code](https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax)
3. **Non-local Attention Learning on Large Heterogeneous Information Networks** *Xiao, Yuxin, Zecheng Zhang, Carl Yang, and Chengxiang Zhai.* IEEE BigData 2019 [paper](https://www.computer.org/csdl/pds/api/csdl/proceedings/download-article/1hJrR2Z6qmA/pdf) [code](https://github.com/xiaoyuxin1002/NLAH)
4. **Multi-Relational Classification via Bayesian Ranked Non-Linear Embeddings** *Rashed, Ahmed, Josif Grabocka, and Lars Schmidt-Thieme.* KDD 2019 [paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330863) [code](https://github.com/ahmedrashed-ml/BRNLE)
5. **RUnimp: SOLUTION FOR KDDCUP 2021 MAG240M-LSC** *Yunsheng Shi, Zhengjie Huang , Weibin Li , Weiyue Su, Shikun Feng* KDD CUP 2021 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_BD-PGL.pdf) [code](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2021/MAG240M/r_unimp)
6. **LARGE-SCALE NODE CLASSIFICATION WITH BOOTSTRAPPING** *Petar Velickovic , Peter Battaglia, Jonathan Godwin, Alvaro Sanchez, David Budden, Shantanu Thakoor, Jacklynn Stott , Ravichandra Addanki , Thomas Keck , Andreea Deac* KDD CUP 2021 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_Academic.pdf) [code](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc/mag)
7. **SYNERISE AT KDD CUP 2021: NODE CLASSIFICATION IN MASSIVE HETEROGENEOUS GRAPHS** *Michal Daniluk , Jacek Dabrowski, Konrad Goluchowski , Barbara Rychalska* KDD CUP 2021 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_SyneriseAI.pdf) [code](https://github.com/Synerise/kdd-cup-2021)
8. **METAPATH-BASED LABEL PROPAGATION FOR LARGE-SCALE HETEROGENEOUS GRAPH** *Qiuying Peng , Wencai Cao, Zheng Pan* KDD CUP 2021 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_Topology_mag.pdf) [code](https://github.com/qypeng-ustc/mplp)
9. **KDD CUP 2021 MAG240M-LSC TEAM PASSAGES WINNER SOLUTION** *Bole Ai , Xiang Long , Kaiyuan Li , Quan Lin , Xiaofan Liu , Pengfei Wang , Mingdao Wang , Zhichao Feng, Kun Zhao* KDD CUP 2021 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_passages.pdf) [code](https://github.com/passages-kddcup2021/KDDCUP2021_OGB_LSC_MAG240M)
10. **DEEPERBIGGERBETTER FOR OGB-LSC AT KDD CUP 2021** *Guohao Li , Hesham Mostafa , Jesus Alejandro Zarzar Torano , Sami Abu-El-Haija, Marcel Nassar, Daniel Cummings , Sohil Shah, Matthias Mueller, Bernard Ghanem* KDD CUP 2020 [paper](https://ogb.stanford.edu/paper/kddcup2021/mag240m_DeeperBiggerBetter.pdf) [code](https://github.com/zarzarj/DeeperBiggerBetter_KDDCup)


## [GNNs for Recommendation](#content)

### Most Influential

1. **Graph  Convolutional Matrix Completion.** *Rianne van den Berg, Thomas N. Kipf, Max Welling*. KDD 2018. [paper](https://arxiv.org/abs/1706.02263v2) [code](https://github.com/riannevdberg/gc-mc)
2. **Session-based recommendation with graph neural networks**. *Shu Wu, Yuyuan Tang, Yanqiao Zhu, Xing Xie, Liang Wang, Tieniu Tan*. AAAI 2019. [paper](https://arxiv.org/abs/1811.00855v1) [code](https://github.com/CRIPAC-DIG/SR-GNN#paper-data-and-code)
3. **Neural Graph Collaborative Filtering**. *Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua*. SIGIR 2019. [paper](https://arxiv.org/abs/1905.08108v1) [code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
4. **Graph Neural Networks for Social Recommendation**.*Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, Dawei Yin*. WWW 2019. [paper](https://export.arxiv.org/pdf/1902.00724) [code](https://github.com/wenqifan03/GraphRec-WWW19)
5. **KGAT: Knowledge Graph Attention Network for Recommendation**. *Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, Tat-Seng Chua*. KDD 2019. [paper](https://arxiv.org/abs/1905.07854v1) [code](https://github.com/xiangwang1223/knowledge_graph_attention_network)
6. **Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction**. *Zekun Li, Zeyu Cui, Shu Wu, Xiaoyu Zhang, Liang Wang*. WWW 2019. [paper](https://arxiv.org/pdf/1910.05552.pdf) [code](https://github.com/CRIPAC-DIG/Fi_GNN)
7. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. SIGIR 2020. [paper](https://arxiv.org/abs/2002.02126v1) [code](https://github.com/kuandeng/LightGCN)
8. **Revisiting  Graph based Collaborative Filtering: A Linear Residual Graph Convolutional  Network Approach**. *Lei Chen, Richang Hong, Kun Zhang, Meng Wang*. AAAI 2020. [paper](https://export.arxiv.org/pdf/2001.10167) [code](https://github.com/newlei/LRGCCF)
9.  **TAGNN:   Target Attentive Graph Neural Networks for Session-based Recommendation**. *Feng Yu, Yanqiao Zhu, Qiang Liu, Shu Wu, Liang Wang, Tieniu Tan*. SIGIR 2020. [paper](https://arxiv.org/pdf/2005.02844.pdf) [code](https://github.com/johnny12150/TA-GNN)
10. **Multi-behavior  Recommendation with Graph Convolutional Networks**. *Jin, Bowen and Gao, Chen and He, Xiangnan and Jin, Depeng and Li, Yong*. SIGIR 2020. [paper](http://staff.ustc.edu.cn/~hexn/papers/sigir20-MBGCN.pdf) [code]()

### Recent SOTA

1. **A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks**. *Sun, Jianing and Guo, Wei and Zhang, Dengcheng and Zhang, Yingxue and Regol, Florence and Hu, Yaochen and Guo, Huifeng and Tang, Ruiming and Yuan, Han and He, Xiuqiang and Coates, Mark*. KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403254) [code]()
2. **MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems.** *Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, Jie Tang*. KDD 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467408) [code](https://github.com/huangtinglin/MixGCF)
3. **Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation.** *Bowen Hao, Jing Zhang, Hongzhi Yin, Cuiping Li, Hong Chen*. WSDM 2021. [paper](https://arxiv.org/pdf/2012.07064v1.pdf) [code](https://github.com/jerryhao66/Pretrain-Recsys)
4. **Self-supervised Graph Learning for Recommendation.** *Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, Xing Xie*. SIGIR 2021. [paper](https://arxiv.org/abs/2010.10783v4) [code](https://github.com/wujcan/SGL)
5. **Interest-aware Message-Passing GCN for Recommendation.** *Fan Liu, Zhiyong Cheng, Lei Zhu, Zan Gao, Liqiang Nie*. WWW 2021. [paper](https://arxiv.org/abs/2102.10044) [code](https://github.com/liufancs/IMP_GCN)
6. **Neural Graph Matching based Collaborative Filtering.** *Yixin Su, Rui Zhang, Sarah Erfani, Junhao Gan*. SIGIR 2021. [paper](https://arxiv.org/abs/2105.04067) [code](https://github.com/ruizhang-ai/GMCF_Neural_Graph_Matching_based_Collaborative_Filtering)
7. **Sequential Recommendation with Graph Convolutional Networks.** *Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, Depeng Jin, Yong Li*. SIGIR 2021. [paper](https://arxiv.org/abs/2106.14226) [code]()
8. **Structured Graph Convolutional Networks with Stochastic Masks for Recommender Systems.** *Chen, Huiyuan and Wang, Lan and Lin, Yusan and Yeh, Chin-Chia Michael and Wang, Fei and Yang, Hao*. SIGIR 2021. [paper](https://dl.acm.org/doi/10.1145/3404835.3462868) [code]()
9. **Self-Supervised Graph Co-Training for Session-based Recommendation.** *Xin Xia, Hongzhi Yin, Junliang Yu, Yingxia Shao, Lizhen Cui*. CIKM 2021. [paper](https://arxiv.org/abs/2108.10560) [code](https://github.com/xiaxin1998/cotrec)
10. **Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph.** *Riku Togashi, Mayu Otani, Shin'ichi Satoh*. WSDM 2021. [paper](https://arxiv.org/pdf/2011.05061.pdf) [code]()


## [Chemistry and Biology](#content)

### Most Influential

1. **Graph u-nets**. *Gao Hongyang, Ji Shuiwang*. international conference on machine learning 2019. [paper](http://proceedings.mlr.press/v97/gao19a.html)
2. **MoleculeNet: a benchmark for molecular machine learning**. *Wu Zhenqin, Ramsundar Bharath, Feinberg Evan N, Gomes Joseph, Geniesse Caleb, Pappu Aneesh S, Leswing Karl, Pande Vijay*. Chemical science 2018. [paper](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a)
3. **An end-to-end deep learning architecture for graph classification**. *Zhang Muhan, Cui Zhicheng, Neumann Marion, Chen Yixin*. ThirtySecond AAAI Conference on Artificial Intelligence 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17146)
4. **Hierarchical graph representation learning with differentiable pooling**. *Ying Rex, You Jiaxuan, Morris Christopher, Ren Xiang, Hamilton William L, Leskovec Jure*. arXiv preprint arXiv:1806.08804 2018. [paper](https://arxiv.org/abs/1806.08804)
5. **How powerful are graph neural networks?**. *Xu Keyulu, Hu Weihua, Leskovec Jure, Jegelka Stefanie*. arXiv preprint arXiv:1810.00826 2018. [paper](https://arxiv.org/abs/1810.00826)
6. **Graph classification using structural attention**. *Lee John Boaz, Rossi Ryan, Kong Xiangnan*. Proceedings of the th ACM SIGKDD International Conference on Knowledge Discovery  Data Mining 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219980)
7. **Neural message passing for quantum chemistry**. *Gilmer Justin, Schoenholz Samuel S, Riley Patrick F, Vinyals Oriol, Dahl George E*. International conference on machine learning 2017. [paper](http://proceedings.mlr.press/v70/gilmer17a)
8. **Learning convolutional neural networks for graphs**. *Niepert Mathias, Ahmed Mohamed, Kutzkov Konstantin*. International conference on machine learning 2016. [paper](http://proceedings.mlr.press/v48/niepert16)
9. **Deep convolutional networks on graph-structured data**. *Henaff Mikael, Bruna Joan, LeCun Yann*. arXiv preprint arXiv:1506.05163 2015. [paper](https://arxiv.org/abs/1506.05163)
10. **Convolutional networks on graphs for learning molecular fingerprints**. *Duvenaud David, Maclaurin Dougal, Aguilera-Iparraguirre Jorge, Gómez-Bombarelli Rafael, Hirzel Timothy, Aspuru-Guzik Alán, Adams Ryan P*. arXiv preprint arXiv:1509.09292 2015. [paper](https://arxiv.org/abs/1509.09292)

### Recent SOTA

1. **Biological network analysis with deep learning**. *Muzio Giulia, O'Bray Leslie, Borgwardt Karsten*. Briefings in bioinformatics 2021. [paper](https://academic.oup.com/bib/article/22/2/1515/5964185?login=true)
2. **Do Transformers Really Perform Bad for Graph Representation?**. *Ying Chengxuan, Cai Tianle, Luo Shengjie, Zheng Shuxin, Ke Guolin, He Di, Shen Yanming, Liu Tie-Yan*. arXiv preprint arXiv:2106.05234 2021. [paper](https://arxiv.org/abs/2106.05234)
3. **Directed acyclic graph neural networks**. *Thost Veronika, Chen Jie*. arXiv preprint arXiv:2101.07965 2021. [paper](https://arxiv.org/abs/2101.07965)
4. **Directional graph networks**. *Beani Dominique, Passaro Saro, Létourneau Vincent, Hamilton Will, Corso Gabriele, Liò Pietro*. International Conference on Machine Learning 2021. [paper](http://proceedings.mlr.press/v139/beani21a.html)
5. **Benchmarking graph neural networks**. *Dwivedi Vijay Prakash, Joshi Chaitanya K, Laurent Thomas, Bengio Yoshua, Bresson Xavier*. arXiv preprint arXiv:2003.00982 2020. [paper](https://arxiv.org/abs/2003.00982)
6. **Structpool: Structured graph pooling via conditional random fields**. *Yuan Hao, Ji Shuiwang*. Proceedings of the th International Conference on Learning Representations 2020. [paper](https://par.nsf.gov/servlets/purl/10159731)
7. **A deep learning approach to antibiotic discovery**. *Stokes Jonathan M, Yang Kevin, Swanson Kyle, Jin Wengong, Cubillos-Ruiz Andres, Donghia Nina M, MacNair Craig R, French Shawn, Carfrae Lindsey A, Bloom-Ackermann Zohar, others*. Cell 2020. [paper](https://www.sciencedirect.com/science/article/pii/S0092867420301021)
8. **Principal neighbourhood aggregation for graph nets**. *Corso Gabriele, Cavalleri Luca, Beaini Dominique, Liò Pietro, Veličković Petar*. arXiv preprint arXiv:2004.05718 2020. [paper](https://arxiv.org/abs/2004.05718)
9. **A fair comparison of graph neural networks for graph classification**. *Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio*. arXiv preprint arXiv:1912.09893 2019. [paper](https://arxiv.org/abs/1912.09893)
10. **Graph convolutional networks with eigenpooling**. *Ma Yao, Wang Suhang, Aggarwal Charu C, Tang Jiliang*. Proceedings of the th ACM SIGKDD International Conference on Knowledge Discovery  Data Mining 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330982)
