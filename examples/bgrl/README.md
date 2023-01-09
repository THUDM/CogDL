# Large-Scale Representation Learning on Graphs via Bootstrapping (BGRL) with CogDL
This is an attempt to implement BGRL with CogDL for graph representation. The authors' implementation can be found [here](https://github.com/nerdslab/bgrl). Another version of the implementation from [Namkyeong](https://github.com/Namkyeong/BGRL_Pytorch) can also be used as a reference.

## Hyperparameters
Some optional parameters are allowed to be added to the training process.

`layers`: the dimension for each layer of GNN.

`pred_hid`: the hidden dimension of the predict moudle.

`aug_params`: the ratio of pollution for graph augmentation.

## Usage
You can find their datasets [here](https://pan.baidu.com/s/15RyvXD2G-xwGM9jrT7IDLQ?pwd=85vv) and put them in the path `./data`. Experiments on their datasets with given hyperparameters can be achieved by the following commands. 

### Wiki-CS
```
python train.py --name WikiCS --aug_params 0.2 0.1 0.2 0.3 --layers 512 256 --pred_hid 512 --lr 0.0001 -epochs 10000 -cs 250 
```
### Amazon Computers
```
python train.py --name computers --aug_params 0.2 0.1 0.5 0.4 --layers 256 128 --pred_hid 512 --lr 0.0005 --epochs 10000 -cs 250
```
### Amazon Photo
```
python train.py --name photo --aug_params 0.1 0.2 0.4 0.1 --layers 512 256 --pred_hid 512 --lr 0.0001 --epochs 10000 -cs 250
```
### Coauthor CS
```
python train.py --name cs --aug_params 0.3 0.4 0.3 0.2 --layers 512 256 --pred_hid 512 --lr 0.00001 --epochs 10000 -cs 250
```
### Coauthor Physics
```
python train.py --name physics --aug_params 0.1 0.4 0.4 0.1 --layers 256 128 --pred_hid 512 --lr 0.00001 --epochs 10000 -cs 250
```

## Performance
The results on five datasets shown on the table.

|          |Wiki-CS|Computers|Photo    |CS   |Physics| 
|------    |------ |---------|---------|-----|-------| 
|Paper     |79.98  |90.34    |93.17    |93.31|95.73  |
|Namkyeong |79.50  |88.21    |92.76    |92.49|94.89  |
|CogDL     |79.76  |88.06    |92.91    |93.05|95.46  |
* Hyperparameters are from original paper

