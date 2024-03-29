# CGCN-CGAT-E
Combining Contextualised Graph Convolution Network and Graph Attention Network with Edge Features for Relation Extraction 

**Preparation**:

Semeval 2010 Task 8 dataset files are available in the directory `dataset/semeval`.
Download and unzip GloVe vectors into directory: dataset. Then prepare vocabulary and initial word vectors with:
```
python3 prepare_vocab.py dataset/semeval dataset/vocab --glove_dir dataset/glove
```
This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

**Training**

To train the CGCN-CGATE model, run:
```
python train.py --id 1 --seed 1 --hidden_dim 360 --lr 0.5 --rnn_hidden 300 --num_epoch 150 --pooling max  --mlp_layers 1 --num_layers 2 --pooling_l2 0.002
```
Model checkpoints and logs will be saved to: `./saved_models/01`.
Please see `train.py` for details of different parameters.

**Evaluation**

Trained modes are saved in the directory: saved_models/01.
To run evaluation on the test set, run:
```
python3 eval.py saved_models/01 --dataset test
```

**References:**

Attention Guided Graph Convolution Networks. https://github.com/Cartus/AGGCN
