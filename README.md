# hrmm4ET

A Hierarchical Hyper-rectangle Mass Model for Fine-grained Entity Typing


## Getting Started 

### Dependencies

```bash

git@github.com:ShingHuuu/hrmm4ET.git

```

This code has been tested with Python 3.8 and the following dependencies:

- `torch==1.10.1` 
- `transformers==4.2.0`


### File Descriptions

- `hrmm/main.py`: Main script for training and evaluating models.
- `hrmm/models.py`: Defines hrmm model.
- `hrmm/modules.py`: Defines hrmm model.
- `hrmm/data_utils.py`: Contains data loader and utility functions.
- `hrmm/constant.py`: Defines paths and constant parameters.
- `hrmm/scorer.py`: Compute precision, recall, and F1 given an output file.
- `hrmm/train_*.sh`: Bash training command.
- `hrmm/eval_*.sh`: Bash evaluation command.
- `hrmm/adaptive_thre.py`: Defines adaptive threshold.

## Datasets / Models

This code assumes 3 directories listed below.
- `./data`: This directory contains original train/dev data files.
- `./data/ontology`: This directory contains types ontology of dataset. 
- `./model`: Trained models will be saved in this directory. 


The data files are formatted as jsonlines. Here is an example of Ontonotes:
```
{"ex_id": "test_5", "right_context": ["."], "left_context": ["The", "broken", "purchase", "appears", "as", "additional", "evidence", "of", "trouble", "at"], "right_context_text": ".", "left_context_text": "The broken purchase appears as additional evidence of trouble at", "y_category": ["/organization", "/organization/company"], "word": "Imperial Corp. , whose spokesman said the company withdrew its application from the federal Office of Thrift Supervision because of an informal notice that Imperial 's thrift unit failed to meet Community Reinvestment Act requirements", "mention_as_list": ["Imperial", "Corp.", ",", "whose", "spokesman", "said", "the", "company", "withdrew", "its", "application", "from", "the", "federal", "Office", "of", "Thrift", "Supervision", "because", "of", "an", "informal", "notice", "that", "Imperial", "'s", "thrift", "unit", "failed", "to", "meet", "Community", "Reinvestment", "Act", "requirements"]}

```

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `ex_id`                   | Unique example ID.                                                                       |
| `right_context`           | Tokenized right context of a mention.                                                    |
| `left_context`            | Tokenized left context of a mention.                                                     |
| `word`                    | A mention.                                                                               |
| `right_context_text`      | Right context of a mention.                                                              |
| `left_context_text`       | Left context of a mention.                                                               |
| `y_category`              | The gold entity types derived from Wikipedia categories.                                 |
| `y_title`                 | Wikipedia title of the gold Wiki entity.                                                 |
| `mention_as_list`         | A tokenized mention.                                                                     |


## Entity Typing Training and Evaluation

### Training

`main.py` is the primary script for training and evaluating models. It starts from bash command `hrmm/train_*.sh`.

```bash
$ cd hrmm
$ bash train_hrmm.sh
```

### Evaluation

```bash
$ cd hrmm
$ bash eval_hrmm.sh
```
