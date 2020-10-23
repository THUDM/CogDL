from cogdl.datasets.kg_data import *

if __name__=="__main__":
    fb15k = FB15KDataset()
    wn18 = WN18Dataset()
    wnrr = WNRRDataset()
    yago3_10 = YAGO310Dataset()
    print(fb15k[:10])
    print(wn18[:10])
    print(wnrr[:10])
    print(yago3_10[:10])