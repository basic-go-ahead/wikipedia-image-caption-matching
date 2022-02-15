# wikipedia-image-caption-matching
This is the 3rd place solution code for [the Wikipedia - Image/Caption Matching Competition on Kaggle](https://www.kaggle.com/c/wikipedia-image-caption).

## [Data Preparation](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation)


### Basic Preprocessing ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/traindataset-part0-4-count-5.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-dataset.ipynb))

![Basic Preprocessing](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/images/dfd-basic-preprocessing.png?raw=true)

### Image Data Preparation

![Image Data Preparation](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/images/dfd-image-data-preparation.png?raw=true)

#### Translating Image Filenames ([train🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/trans-fn-traindataset-part0-4-count-5-part5-5.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-trans-filename.ipynb))

#### Calculating SEs for original filenames ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/undigit-filename-sentence-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-undigit-filename-sentence-embeddings.ipynb))

#### Forming Final Image Dataset ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/final-image-traindataset-part0-4-count-5.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-final-image-dataset.ipynb))

#### Calculating SEs for translated filenames ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/translated-filename-sentence-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-translated-filename-sentence-embeddings.ipynb))

#### Calculating ITEs for images ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/train-image-embeddings-sizes.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-image-embeddings-sizes.ipynb))

### Candidate Data Preparation

![Candidate Data Preparation](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/images/dfd-candidate-data-preparation.png?raw=true)

#### Translating page titles ([train🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/trans-title-traindataset0-5-part0-21.ipynb) | [test🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-trans-page-title-1-6.ipynb))

#### Translating captions ([train🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/trans-cap-traindataset0-5-part21-24.ipynb) | [test🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-trans-caption-1-6.ipynb))

#### Forming Final Candidate Dataset ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/combiner-target-dataset-0-of-5.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-combiner-target-dataset.ipynb))

#### Calculating Caption SEs ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/caption-sentence-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-caption-sentence-embeddings.ipynb))

#### Calculating Page Title SEs ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/pagetitle-sentence-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-pagetitle-sentence-embeddings.ipynb))

#### NER for captions ([train🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/caption-original-ner-part1-3.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-caption-final-ner.ipynb))

#### NER for page titles ([train🧩](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/title-final-ner-part1-3.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-title-final-ner.ipynb))

#### Calculating ITEs for original candidates ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/train-undigit-target-512-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-undigit-target-512-embeddings.ipynb))

#### Calculating ITEs for translated candidates ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/train-final-target-512-embeddings.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-final-target-512-embeddings.ipynb))

#### Building Containers ([train](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/train/train-target-containers.ipynb) | [test](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/data_preparation/test/test-target-containers.ipynb))



## [Filtering](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/filtering)

## [Ranking](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking)

Now, for each image, the matching problem comes to ranking its candidates with `XGBRanker`.

After filtering, the data prepared is split into training dataset, validation dataset, and holdout dataset as follows.

| Dataset    | Parts       |
| ---------- | ----------- |
| Training   | 0–58, 60–68 |
| Validation | 70, 71      |
| Holdout    | 59, 69      |

According to the table below, the training dataset, in turn, is divided into 7 ranges, each intended for training a certain base model.

| Base Model | Part Range |
| ---------- | ---------- |
| `model-00` | 0–9        |
| `model-01` | 10–19      |
| `model-02` | 20–29      |
| `model-03` | 30–39      |
| `model-04` | 40–49      |
| `model-05` | 50–58      |
| `model-06` | 60–68      |

In accordance with stacking techniques, the final model is trained on the ranks made by the base models.

The training and inference procedures are depicted visually in the following diagram.

![The training and inference procedures](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/images/dfd-ranking.png?raw=true)

### Training Pipeline

#### Preprocessing ([data-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/base_models/data-for-model-XX.ipynb))

This notebook makes final preparations of filtered data forming datasets for base models.

The notebook should serve as a template to produce training data for a specified range of parts expressed by the variable `splits`.

#### Training Base Models ([train-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/base_models))

Fits `XGBRanker` to a specified training dataset. These notebooks differ only in hyperparameters and the path to the folder containing tranining data.

#### Calculating Ranks ([train-ranks-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/train-ranks-for-model-XX.ipynb))

Uses `model-00` to produce its ranks for the validation and holdout datasets.
For each image, the ranks obtained are used to determine the 50 candidates with the highest rank, while the rest of the ones are rejected.
The best candidates and its ranks subsequently form training and validation data for the final model.

The notebook should serve as a template to produce ranks for the rest base models. Just set a suitable value for `MODEL_PATH` in order to switch the notebook to another base model.

#### Rank Stacking ([data-for-final-model.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/data-for-final-model.ipynb))

The ranks computed in the previous stage are stacked.

#### Final Model Training ([train-final-model.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/train-final-model.ipynb))

### Inference Pipeline

#### Preprocessing and Calculating Ranks ([test-ranks-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/test-ranks-for-model-XX.ipynb))

Uses `model-00` to produce its ranks for the test dataset.

The notebook should serve as a template to produce ranks for the rest base models. Just set a suitable value for `MODEL_PATH` in order to switch the notebook to another base model.