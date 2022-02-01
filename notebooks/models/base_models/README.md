## `data-for-model-XX.ipynb`

This notebook makes final preparations of filtered data forming training data for base models.

The notebook should serve as a template to produce training data for a specified range of parts expressed by the variable `splits`. Each range results in a training dataset meant for a corresponding base model.

## `train-model-XX.ipynb`

Fits `XGBRanker` to a specified training dataset. These notebooks differ only in 