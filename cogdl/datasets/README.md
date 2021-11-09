Datasets of CogDL
=================

CogDL now supports the following datasets for different tasks:
- Network Embedding (Unsupervised node classification): PPI, Blogcatalog, Wikipedia, Youtube, DBLP, Flickr
- Semi/Un-superviesd Node classification: Cora, Citeseer, Pubmed, Reddit, PPI, PPI-large, Yelp, Flickr, Amazon
- Heterogeneous node classification: DBLP, ACM, IMDB
- Link prediction: PPI, Wikipedia, Blogcatalog
- Multiplex link prediction: Amazon, YouTube, Twitter
- graph classification: MUTAG, IMDB-B, IMDB-M, PROTEINS, COLLAB, NCI, NCI109, Reddit-BINARY

<h3>Node classification</h3>


<table>
    <tr>
        <th></th>
    	<th>Dataset</th>
        <th>#Nodes</th>
        <th>#Edges</th>
        <th>#Features</th>
        <th>#Classes</th>
        <th>#Train/Val/Test</th>
        <th>Degree</th>
        <th>#Name in Cogdl</th>
    </tr>
    <tr>
    	<th rowspan="10">Transductive</th>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/71ee1916f3644f6f81b4/"> Cora </a> </td>
        <td> 2,708 </td>
        <td> 5,429 </td>
        <td> 1,433 </td>
        <td> 7(s) </td>
        <td> 140 / 500 / 1000 </td>
        <td> 2 </td>
        <td> cora </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/6823f768780e460e960b/"> Citeseer </a> </td>
        <td> 3,327 </td>
        <td> 4,732 </td>
        <td> 3,703 </td>
        <td> 6(s) </td>
        <td> 120 / 500 / 1000 </td>
        <td> 1 </td>
        <td> citeseer </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/1b2f95b3d392463bb024/"> PubMed </a> </td>
        <td> 19,717 </td>
        <td> 44,338 </td>
        <td> 500 </td>
        <td> 3(s) </td>
        <td> 60 / 500 / 1999 </td>
        <td> 2 </td>
        <td> pubmed </td>
    </tr>
    <tr>
        <td> <a href=""> Chameleon </a> </td>
        <td> 2,277 </td>
        <td> 36,101 </td>
        <td> 2,325 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 16 </td>
        <td> chameleon </td>
    </tr>
        <tr>
        <td> <a href=""> Cornell </a> </td>
        <td> 183 </td>
        <td> 298 </td>
        <td> 1,703 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 1.6 </td>
        <td> cornell </td>
    </tr>
    </tr>
        <td> <a href=""> Film </a> </td>
        <td> 7,600 </td>
        <td> 30,019 </td>
        <td> 932 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 4 </td>
        <td> film </td>
    </tr>
    </tr>
        <td> <a href=""> Squirrel </a> </td>
        <td> 5201 </td>
        <td> 217,073 </td>
        <td> 2,089 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 41.7 </td>
        <td> squirrel </td>
    </tr>
    </tr>
        <td> <a href=""> Texas </a> </td>
        <td> 182 </td>
        <td> 325 </td>
        <td> 1,703 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 1.8 </td>
        <td> texas </td>
    </tr>
        </tr>
        <td> <a href=""> Wisconsin </a> </td>
        <td> 251 </td>
        <td> 515 </td>
        <td> 1,703 </td>
        <td> 5 </td>
        <td> 0.48 / 0.32 / 0.20 </td>
        <td> 2 </td>
        <td> Wisconsin </td>
    </tr>
    <tr>
        <th rowspan="8"> Inductive </th>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/2c6e94ec9dad4972b58e/"> PPI </a> </td>
        <td> 14,755 </td>
        <td> 225,270 </td>
        <td> 50 </td>
        <td> 121(m) </td>
        <td> 0.66 / 0.12 / 0.22 </td>
        <td> 15 </td>
        <td> ppi </td>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/469ef38a520640bba267/"> PPI-large </a>  </td>
        <td> 56,944 </td>
        <td> 818,736 </td>
        <td> 50 </td>
        <td> 121(m) </td>
        <td> 0.79 / 0.11 / 0.10 </td>
        <td> 14 </td>
        <td> ppi-large </td>
    </tr>
    <tr>
        <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/d087e7e766e747ce8073/"> Reddit </a>  </td>
        <td> 232,965 </td>
        <td> 11,606,919 </td>
        <td> 602 </td>
        <td> 41(s) </td>
        <td> 0.66 / 0.10 / 0.24 </td>
        <td> 50 </td>
        <td> reddit </td>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/d3ebcb5fa2da463b8213/"> Flickr </a>  </td>
        <td> 89,250 </td>
        <td> 899,756 </td>
        <td> 500 </td>
        <td> 7(s) </td>
        <td> 0.50 / 0.25 / 0.25 </td>
        <td> 10 </td>
        <td> flickr </td>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/7218cc013c9a40159306/"> Yelp </a> </td>
        <td> 716,847 </td>
        <td> 6,977,410 </td>
        <td> 300 </td>
        <td> 100(m) </td>
        <td> 0.75 / 0.10 / 0.15 </td>
        <td> 10 </td>
        <td> yelp </td>
    </tr>
    <tr>
        <td> <a href="https://cloud.tsinghua.edu.cn/d/ae4b2c4f59bd41be9b0b/"> Amazon-SAINT </a> </td>
        <td> 1,598,960 </td>
        <td> 132,169,734 </td>
        <td> 200 </td>
        <td> 107(m) </td>
        <td> 0.85 / 0.05 / 0.10 </td>
        <td> 83 </td>
        <td> amazon-s </td>
    </tr>
</table>


<h3>Network Embedding(Unsupervised Node classification)</h3>


<table>
    <tr>
    	<th>Dataset</th>
        <th>#Nodes</th>
        <th>#Edges</th>
        <th>#Classes</th>
        <th>#Degree</th>
        <th>#Name in Cogdl</th>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/fd3717d9ee78440e800f/"> PPI </a> </td>
        <td> 3,890 </td>
        <td> 76,584 </td>
        <td> 50(m) </td>
        <td> 20 </td>
        <td> ppi-ne </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/cb62b5b4224a4de08a02/"> BlogCatalog </a> </td>
        <td> 10,312 </td>
        <td> 333,983 </td>
        <td> 40(m) </td>
        <td> 32 </td>
        <td> blogcatalog </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/a26619b0b45e4d1181c9/"> Wikipedia </a>  </td>
        <td> 4.777 </td>
        <td> 184,812 </td>
        <td> 39(m) </td>
        <td> 39 </td>
        <td> wikipedia </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/863da94f520844cbab90/"> Flickr </a> </td>
        <td> 80,513 </td>
        <td> 5,899,882 </td>
        <td> 195(m) </td>
        <td> 73 </td>
        <td> flickr-ne </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/1da2ec50b08749f48033/"> DBLP </a> </td>
        <td> 51,264 </td>
        <td> 2,990,443 </td>
        <td> 60(m) </td>
        <td> 2 </td>
        <td> dblp-ne </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/e338d719659b44e5ac9d/"> Youtube </a>  </td>
        <td> 1,138,499 </td>
        <td> 2,990,443 </td>
        <td> 47(m) </td>
        <td> 3 </td>
        <td> youtube-ne </td>
    </tr>
</table>


<h3>Heterogenous Graph</h3>


<table>
    <tr>
    	<th>Dataset</th>
        <th>#Nodes</th>
        <th>#Edges</th>
        <th>#Features</th>
        <th>#Classes</th>
        <th>#Train/Val/Test</th>
        <th>#Degree</th>
        <th>#Edge Type</th>
        <th>#Name in Cogdl</th>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/f/f15a18f34c084a7c9482/?dl=1"> DBLP </a>  </td>
        <td> 18,405 </td>
        <td> 67,946 </td>
        <td> 334 </td>
        <td> 4 </td>
        <td> 800 / 400 / 2857 </td>
        <td> 4 </td>
        <td> 4 </td>
        <td> gtn-dblp(han-acm) </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/f/5d8f1290ea3946edaec2/?dl=1"> ACM </a>  </td>
        <td> 8,994 </td>
        <td> 25,922 </td>
        <td> 1,902 </td>
        <td> 3 </td>
        <td> 600 / 300 / 2125 </td>
        <td> 3 </td>
        <td> 4 </td>
        <td> gtn-acm(han-acm) </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/f/0617f97635134505bb1c/?dl=1"> IMDB </a>  </td>
        <td> 12,772 </td>
        <td> 37,288 </td>
        <td> 1,256 </td>
        <td> 3 </td>
        <td> 300 / 300 / 2339 </td>
        <td> 3 </td>
        <td> 4 </td>
        <td> gtn-imdb(han-imdb) </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/21e0ad8cfe564bc3b17a/"> Amazon-GATNE </a> </td>
        <td> 10,166 </td>
        <td> 148,863 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 15 </td>
        <td> 2 </td>
        <td> amazon </td>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/d/55a23e4edba54c29a7c2/"> Youtube-GATNE </a>  </td>
        <td> 2,000 </td>
        <td> 1,310,617 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 655 </td>
        <td> 5 </td>
        <td> youtube </td>
    </tr>
    <tr>
    	<td> <a href="(https://cloud.tsinghua.edu.cn/d/59b52be66cbf4d20a414/"> Twitter </a>  </td>
        <td> 10,000 </td>
        <td> 331,899 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 33 </td>
        <td> 4 </td>
        <td> twitter </td>
    </tr>
</table>


<h3>Knowledge Graph Link Prediction</h3>


<table>
    <tr>
   		<th>Dataset</th>
        <th>#Nodes</th>
        <th>#Edges</th>
        <th>#Train/Val/Test</th>
        <th>#Relations Types</th>
        <th>#Degree</th>
        <th>#Name in Cogdl</th>
    </tr>
    <tr>
    	<td><a href="https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB13"> FB13 </a> </td>
        <td>75,043</td>
        <td>345,872</td>
        <td>316,232 / 5,908 / 23,733</td>
        <td>12</td>
        <td>5</td>
        <td>fb13</td>
    </tr>
    <tr>
    	<td><a href="https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB15K"> FB15k </a> </td>
        <td>14,951</td>
        <td>592,213</td>
        <td>483,142 / 50,000 / 59,071</td>
        <td>1345</td>
        <td>40</td>
        <td>fb15k</td>
    </tr> 
    <tr>
    	<td><a href="https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB15K237"> FB15k-237 </a> </td>
        <td>14,541</td>
        <td>310,116</td>
        <td>272,115 / 17,535 / 20,466</td>
        <td>237</td>
        <td>21</td>
        <td>fb15k237</td>
    </tr>
    <tr>
    	<td><a href="https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18"> WN18 </a> </td>
        <td>40,943</td>
        <td>151,442</td>
        <td>141,442 / 5,000 / 5,000</td>
        <td>18</td>
        <td>4</td>
        <td>wn18</td>
    </tr>
    <tr>
    	<td><a href="https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18RR"> WN18RR </a></td>
        <td>86,835</td>
        <td>93,003</td>
        <td>86,835 / 3,034 / 3,134</td>
        <td>11</td>
        <td>1</td>
        <td>wn18rr</td>
    </tr>
</table>


<h3>Graph Classification</h3>

[TUdataset](https://cloud.tsinghua.edu.cn/d/878208c0acf74919959a/) from https://www.chrsmrrs.com/graphkerneldatasets

<table>
    <tr>
    	<th>Dataset</th>
        <th>#Graphs</th>
        <th>#Classes</th>
        <th>#Avg. Size</th>
        <th>#Name in Cogdl</th>
    </tr>
    <tr>
    	<td> <a href="https://cloud.tsinghua.edu.cn/f/f5584198ded14c58b94b/?dl=1"> MUTAG </a></td>
        <td>188</td>
        <td>2</td>
        <td>17.9</td>
        <td>mutag</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/be48c1fafde84569813a/?dl=1"> IMDB-B </a> </td>
        <td>1,000</td>
        <td>2</td>
        <td>19.8</td>
        <td>imdb-b</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/8eca3b50a2094178b2ec/?dl=1"> IMDB-M </a> </td>
        <td>1,500</td>
        <td>3</td>
        <td>13</td>
        <td>imdb-m</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/26654be1c3c946388a56/?dl=1"> PROTEINS </a> </td>
        <td>1,113</td>
        <td>2</td>
        <td>39.1</td>
        <td>proteins</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/c58b948654e54c3d8be7/?dl=1"> COLLAB </a> </td>
        <td>5,000</td>
        <td>5</td>
        <td>508.5</td>
        <td>collab</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/151ea45c7f3444a39537/?dl=1"> NCI1 </a> </td>
        <td>4,110</td>
        <td>2</td>
        <td>29.8</td>
        <td>nci1</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/b520b63bcf9d48a7a57c/?dl=1"> NCI109 </a> </td>
        <td>4,127</td>
        <td>2</td>
        <td>39.7</td>
        <td>nci109</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/9fc07402515549d1b6a6/?dl=1"> PTC-MR </a> </td>
        <td>344</td>
        <td>2</td>
        <td>14.3</td>
        <td>ptc-mr</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/d4bcb32cf6a846f8b7cb/?dl=1"> REDDIT-BINARY </a> </td>
        <td>2,000</td>
        <td>2</td>
        <td>429.7</td>
        <td>reddit-b</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/e9bed9b1181246b7859f/?dl=1"> REDDIT-MULTI-5k </a> </td>
        <td>4,999</td>
        <td>5</td>
        <td>508.5</td>
        <td>reddit-multi-5k</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/f/f1b3ffb83fd04c89be7c/?dl=1"> REDDIT-MULTI-12k </a> </td>
        <td>11,929</td>
        <td>11</td>
        <td>391.5</td>
        <td>reddit-multi-12k</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/d/9db9e16a949b4877bb4e/"> BBBP </a> </td>
        <td>2,039</td>
        <td>2</td>
        <td>24</td>
        <td>bbbp</td>
    </tr>
    <tr>
    	<td><a href="https://cloud.tsinghua.edu.cn/d/c6bd3405569b4fab9c4a/"> BACE </a></td>
        <td>1,513</td>
        <td>2</td>
        <td>34.1</td>
        <td>bace</td>
    </tr>
</table>
