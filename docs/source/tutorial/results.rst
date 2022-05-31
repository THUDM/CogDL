Experimental Results 
====================

CogDL provides several downstream tasks including node classification and graph classification to evaluate implemented methods. We also build a reliable leaderboard for each task, which maintain benchmarks and state-of-the-art results on this task. 


Network embedding
-----------------
Unsupervised node classification task aims to learn a mapping function that projects each node to a d-dimensional space in an unsupervised manner. Structural properties of the network should be captured by the mapping function.
We build a leaderboard for the unsupervised multi-label node classification setting. 
We run all algorithms on several real-world datasets and report the sorted experimental Micro-F1 results (%) using logistic regression with L2 normalization. 
The following table shows the results.

 ======= ==================================================== ================ ================ ================ ================ ================ 
  Rank    Method                                               PPI              Wikipedia        Blogcatalog      DBLP             Flickr          
 ======= ==================================================== ================ ================ ================ ================ ================ 
  1       NetMF (Qiu et al, WSDM'18)                           23\.73 ± 0.22    57\.42 ± 0.56    42\.47 ± 0.35    56\.72 ± 0.14    36\.27 ± 0.17   
  2       ProNE (Zhang et al, IJCAI'19)                        24\.60 ± 0.39    56\.06 ± 0.48    41\.14 ± 0.26    56\.85 ± 0.28    36\.56 ± 0.11   
  3       NetSMF (Qiu et at, WWW'19)                           23\.88 ± 0.35    53\.81 ± 0.58    40\.62 ± 0.35    59\.76 ± 0.41    35\.49 ± 0.07   
  4       Node2vec (Grover et al, KDD'16)                      20\.67 ± 0.54    54\.59 ± 0.51    40\.16 ± 0.29    57\.36 ± 0.39    36\.13 ± 0.13   
  5       LINE (Tang et al, WWW'15)                            21\.82 ± 0.56    52\.46 ± 0.26    38\.06 ± 0.39    49\.78 ± 0.37    31\.61 ± 0.09   
  6       DeepWalk (Perozzi et al, KDD'14)                     20\.74 ± 0.40    49\.53 ± 0.54    40\.48 ± 0.47    57\.54 ± 0.32    36\.09 ± 0.10   
  7       Spectral (Tang et al, Data Min Knowl Disc (2011))    22\.48 ± 0.30    49\.35 ± 0.34    41\.41 ± 0.34    43\.68 ± 0.58    33\.09 ± 0.07   
  8       Hope (Ou et al, KDD'16)                              21\.43 ± 0.32    54\.04 ± 0.47    33\.99 ± 0.35    56\.15 ± 0.22    28\.97 ± 0.19   
  9       GraRep (Cao et al, CIKM'15)                          20\.60 ± 0.34    54\.37 ± 0.40    33\.48 ± 0.30    52\.76 ± 0.42    31\.83 ± 0.12   
 ======= ==================================================== ================ ================ ================ ================ ================ 


Graph neural networks
---------------------
This task is for node classification with GNNs in semi-supervised and self-supervised settings. Different from the previous part, nodes in these graphs, like Cora and Reddit, have node features and are fed into GNNs with prediction or representation as output. Cross-entropy loss and contrastive loss are set for semi-supervised and self-supervised settings, respectively. For evaluation, we use prediction accuracy for multi-class and micro-F1 for multi-label datasets.


======= ================================================ ============== ============== ============== 
Rank    Method                                           Cora           Citeseer       Pubmed        
======= ================================================ ============== ============== ============== 
1       Grand(Feng et al., NIPS'20)                      84\.8 ± 0.3    75\.1 ± 0.3    82\.4 ± 0.4   
2       GCNII(Chen et al., ICML'20)                      85\.1 ± 0.3    71\.3 ± 0.4    80\.2 ± 0.3   
3       DR-GAT (Zou et al., 2019)                        83\.6 ± 0.5    72\.8 ± 0.8    79\.1 ± 0.3   
4       MVGRL (Hassani et al., KDD'20)                   83\.6 ± 0.2    73\.0 ± 0.3    80\.1 ± 0.7   
5       APPNP (Klicpera et al., ICLR'19)                 84\.3 ± 0.8    72\.0 ± 0.2    80\.0 ± 0.2   
6       Graph U-Net (Gao et al., 2019)                   83\.3 ± 0.3    71\.2 ± 0.4    79\.0 ± 0.7   
7       GAT (Veličković et al., ICLR'18)                 82\.9 ± 0.8    71\.0 ± 0.3    78\.9 ± 0.3   
8       GDC\_GCN (Klicpera et al., NeurIPS'19)           82\.5 ± 0.4    71\.2 ± 0.3    79\.8 ± 0.5   
9       DropEdge(Rong et al., ICLR'20)                   82\.1 ± 0.5    72\.1 ± 0.4    79\.7 ± 0.4   
10      GCN (Kipf et al., ICLR'17)                       82\.3 ± 0.3    71\.4 ± 0.4    79\.5 ± 0.2   
11      DGI (Veličković et al., ICLR'19)                 82\.0 ± 0.2    71\.2 ± 0.4    76\.5 ± 0.6   
12      JK-net (Xu et al., ICML'18)                      81\.8 ± 0.2    69\.5 ± 0.4    77\.7 ± 0.6   
13      GraphSAGE (Hamilton et al., NeurIPS'17)          80\.1 ± 0.2    66\.2 ± 0.4    77\.2 ± 0.7   
14      GraphSAGE(unsup)(Hamilton et al., NeurIPS'17)    78\.2 ± 0.9    65\.8 ± 1.0    78\.2 ± 0.7   
15      Chebyshev (Defferrard et al., NeurIPS'16)        79\.0 ± 1.0    69\.8 ± 0.5    68\.6 ± 1.0   
16      MixHop (Abu-El-Haija et al., ICML'19)            81\.9 ± 0.4    71\.4 ± 0.8    80\.8 ± 0.6   
======= ================================================ ============== ============== ============== 

