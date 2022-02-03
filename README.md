# wikipedia-image-caption-matching
This is the 3rd place solution code for [the Wikipedia - Image/Caption Matching Competition on Kaggle](https://www.kaggle.com/c/wikipedia-image-caption).

## Data Preparation

## Filtering

## Ranking

Now, for each image, the matching problem comes to ranking its candidates with `XGBRanker`.

After filtering, the data prepared is split into training dataset, validation dataset, and holdout dataset as follows.

| Dataset    | Parts       |
| ---------- | ----------- |
| Training   | 0–58, 60–68 |
| Validation | 70, 71      |
| Holdout    | 59, 69      |

According to the table below, the training dataset, in turn, is divided into seven ranges, each intended for training a certain base model.

| Base Model | Part Range |
| ---------- | ---------- |
| `model-00` | 0–9        |
| `model-01` | 10–19      |
| `model-02` | 20–29      |
| `model-03` | 30–39      |
| `model-04` | 40–49      |
| `model-05` | 50–58      |
| `model-06` | 60–68      |


![Training and inference pipelines](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/images/dfd-ranking.png?raw=true)<center>*Training and inference pipelines*</center>

### Training Pipeline

#### Preprocessing ([data-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/base_models/data-for-model-XX.ipynb))

This notebook makes final preparations of filtered data forming datasets for base models.

The notebook should serve as a template to produce training data for a specified range of parts expressed by the variable `splits`. Each range results in a training dataset meant for a corresponding base model.

#### Training Base Models ([train-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/base_models))

Fits `XGBRanker` to a specified training dataset. These notebooks differ only in hyperparameters and the path to the folder containing tranining data.

#### Calculating Ranks ([train-ranks-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/train-ranks-for-model-XX.ipynb))

Uses `model-00` to produce its ranks for the validation and holdout datasets. These ranks form training and validation data for the final model.

The notebook should serve as a template to produce ranks for the rest base models. Just set a suitable value for `MODEL_PATH`.

#### Rank Stacking ([data-for-final-model.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/data-for-final-model.ipynb))

#### Final Model Training ([train-final-model.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/train-final-model.ipynb))

### Inference Pipeline

#### Preprocessing and Calculating Ranks ([test-ranks-for-model-XX.ipynb](https://github.com/basic-go-ahead/wikipedia-image-caption-matching/blob/main/notebooks/ranking/final_model/test-ranks-for-model-XX.ipynb))

Uses `model-00` to produce its ranks for the test dataset.

The notebook should serve as a template to produce ranks for the rest base models. Just set a suitable value for `MODEL_PATH`.