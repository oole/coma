# Tensorflow 2 implementation of the Convolutional Mesh Autoencoder (CoMA)
Re-implementation of the Convolutional Mesh Autoencoder ([CoMA](https://github.com/anuragranj/coma)) proposed by Ranjan et al. in their paper [Generating 3D faces using Convolutional Mesh Autoencoders](https://arxiv.org/abs/1807.10267).

## Data
The data for training and evaluation the models is available on the [project page](https://coma.is.tue.mpg.de/) of the original paper.

## Training

The utility

```
./train_models.sh
```
can be used to train the models used for evaluation.

## Evaluation

The evaluation utility is split, so that prediction and error calculation can be done on different computation instances.

To predict the test data using the trained models use:
```
./calculate_predictions.sh
```

To calculate the errors given the prediction output of the above utility, use:
```
./calculate_errors.sh
```

## Main.py
The autoencoders main file can be invoked from commandline via the main.py:
```
python main.py --name run-name --data-folder /path/to/preprocessed/data --mode train|test|latent
```

Output of `python main.py --help`:
```
Convolutional Mesh Autoencoder written for Tensorflow 2

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of the run (used for checkpoints and tensorboard
  --data-dir DATA_DIR   Path to the data folder containing train.npy and test.npy
  --batch-size BATCH_SIZE
                        The batch size to be used (default is 16)
  --num-epochs NUM_EPOCHS
                        The number of training epochs (default is 300
  --initial-epoch INITIAL_EPOCH
                        The initial epoch, useful for continue training on an existing run (default 0)
  --latent-vector-length LATENT_VECTOR_LENGTH
                        The size of the latent vector (default is 8)
  --validation-frequency VALIDATION_FREQUENCY
                        The validation frequency (default is 10)
  --learning-rate LEARNING_RATE
                        The learning rate (default is 8e-3
  --random-seed RANDOM_SEED
                        The random seed (default is 8)
  --template-mesh TEMPLATE_MESH
                        Path to the template mesh (default is data/template.obj)
  --mode MODE           The mode to run in train|test|latent (default is train)
  --sanity-check SANITY_CHECK
                        Whether or not sanity check should be performed (default is False)
  --coma-model-dir COMA_MODEL_DIR
                        The directory holding checkpoints and tensorboard, such as coma-model/tensorboard or coma-model/checkpoint (default is coma-model)
  --visualize-during-training VISUALIZE_DURING_TRAINING
                        Whether the meshes should be visualized in tensorboard during training (default is False)
  --page-through PAGE_THROUGH
                        Whether the test meshes should be opened in an interactive session (default is False
  --result-dir RESULT_DIR
                        The results directory for the tests (default is results)

```
##
Based on: Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, and Michael J. Black. "Generating 3D faces using Convolutional Mesh Autoencoders." European Conference on Computer Vision (ECCV) 2018.
