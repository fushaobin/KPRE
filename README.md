# KPRE

This repository is the implementation of KPRE :

> Knowledge Graph Attention Network with Path Rotate Encoding for Recommendation
>
> 

## Required packages

The code has been tested running under Python 3.7.13, with the following packages installed (along with their dependencies):
- torch==1.12.1
- numpy==1.21.5
- tqdm==4.64.1
- scikit-learn==0.24.1

## Files in the folder

- `data/`
  - `music/` 
    - `ratings_final.txt`: user-item interaction graph of Last.FM dataset;
    - `kg_final.txt`: knowledge graph file;
  - `book/` ( the structure of other datasets is similar )
  - `movie/`
  - `restaurant/`
- `src/`: implementations of KPRE.

## Download  & preprocess data

We have downloaded and processed the `music` and `book` datasets. However, for larger datasets (`Movies` and `Restaurants`) you will need to download and pre-process them yourself. The datasets are pre-processed using the same way as `CKAN`. The following are the original files of the datasets for download:

- Music

```
$ url http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
```


- Book
```
$ url http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
```

- movie
```
$ url http://files.grouplens.org/datasets/movielens/ml-20m.zip
```

- Restaurant
```
$ url https://github.com/hwwang55/KGNN-LS/raw/master/data/restaurant/Dianping-Food.zip
```


##  Run the code

We set a random seed to facilitate users to observe the effect of the model easily. You can reset the random seed by adding parameters this way:  `--random_flag False`

- music

```
$ cd src
$ python main.py --dataset music --adj_size 8 --n_layer 3 --aim_num 3 --lr 0.03
```

- book 

```
$ cd src
$ python main.py --dataset book --adj_size 32 --n_layer 4 --aim_num 4 --lr 0.01
```

- movie

```
$ cd src
$ python main.py --dataset movie --adj_size 64 --n_layer 2 --aim_num 3 --lr 0.01
```

- restaurant

```
$ cd src
$ python main.py --dataset restaurant --adj_size 8 --n_layer 3 --aim_num 3 --lr 0.01
```

