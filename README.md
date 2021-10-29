# rgclass

*rgclas* is a didactic implementation of algorithm proposed by [Nada de Lara and Edourard Pineau](https://arxiv.org/abs/1810.09155).


## Quick Start

If you want to use *rgclass* from linux, the easiest way is to download the compiled version.

### File format

*rgclass* is capable of load a set of networks and their class in the format proposed by this
[repository](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/).

There are 3 files

- **DS_A.txt**: in this file there are all the edges for all the networks of the dataset
- **DS_graph_indicator**: the value in the i-th line is the graph_id of the node with node_id i
- **DS_graph_labels.txt**: the value in the i-th line is the class label of the graph with graph_id i

The folder [dataset](https://github.com/AlessandroBregoli/rgclass/tree/main/datasets) contains 4 of
the dataset used by the author of this model for validation.


### Cross Validation

The program *rgclass* allow to evaluate the effectiveness of the method presented by 
[Nada de Lara and Edourard Pineau](https://arxiv.org/abs/1810.09155) computing the accuracy in
cross validation.


The parameters required to accomplish this task are:

- *-a datasets/MUTAG/MUTAG_A.txt*: path of the adjacency list file in csv format
- *-n datasets/MUTAG/MUTAG_graph_indicator.txt*: path of graph identifiers file in csv format for all nodes of all graphs
- *-g datasets/MUTAG/MUTAG_graph_labels.txt*: path of the class labels for all graphs in the dataset in csv format
- *cross_validate*: execute cross validation with a decision tree classifier
- *-k 10*: number of folds to apply
- *-f 18*: number of features to extract from each graph



```
$ rgclass  -a datasets/MUTAG/MUTAG_A.txt -n datasets/MUTAG/MUTAG_graph_indicator.txt -g datasets/MUTAG/MUTAG_graph_labels.txt cross_validate -k 10 -f 18
Accuracy: 0.8666666746139526
```



## Build from source

To build *rgclass* from source [cargo](https://www.rust-lang.org/tools/install) is
required.

To build *rgclass* from source is also required a blas implementation such as [LAPACK](http://www.netlib.org/lapack/#_software)


Download or clone the current repository. Then execute the following command:

```
cargo build --release
```
The compiled program will be available at: *target/release/rgclass*.
