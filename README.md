# Cancer Pathology

This branch contains NLP models for extracting cancer pathology sentences in medical notes. 

## Prerequisites
- Python: 3.8
- torch: 1.8.1+cu111
- nltk: 3.5
- transformers: 4.8.2
- pandas: 1.3.4

### Structure of the code
```
README.md
SenTag                          # contains code for training the model
   |-- data                     # Store the training and testing dataset
   |   |-- ehr              
   |   |   |-- train.csv
   |   |   |-- test.csv
   |   |   |-- cancer.csv
   |   |   |-- cancer_sep.csv
   |   |   |-- keyword.txt      # Rule-based keywords list 
   |-- src                      # source code for model training
   |   |-- config.py            # model config file used for loading parameters for inference
   |   |-- data_loader.py       # load the training dataset to convert the features
   |   |-- data_processor.py    # data processing for feature converting
   |   |-- evaluate.py          # evaluate the model performacne
   |   |-- inference.py         # main code for model inference used web demo not by production
   |   |-- main.py              # main function for training ad evaluating the model
   |   |-- models               # core model files including layers and model
   |   |   |-- crf.py 
   |   |   |-- layers.py
   |   |   |-- sentag.py
   |   |-- trainer.py           # functions for model training and optmization
   |   |-- utils.py             # utility functions
SenTag_serving                  # constains code for model serving
   |-- config.py                # model config file used for loading parameters for model serving
   |-- data                     # store the necessary data for model serving  
   |   |-- eval_data  
   |   |   |-- ***.json         # put all json files as input here
   |   |-- keyword.txt      
   |-- inference.py             # main code for model inference used by production
   |-- models                   # The model codes under this fodler are same in the model_training
   |   |-- crf.py
   |   |-- layers.py
   |   |-- sentag.py
   |-- outputs                  # store output files if necessary
   |   |-- .gitkeep
   |   |-- pytorch_model.bin                 # put the model here
   |-- run_inference.py         # example file to run the inference code
   |-- utils.py
docs                            # Store some documents about the model
   |-- cancer_pathology_report.docx
```

### Preparing Training Dataset
The dataset is located at the folder /SenTag/data/ehr. The training input data should be a csv containing 4 columns: id, patient_id, note, pathology_alone, the cancer pathology sentence should be separated by <start> and <end> as shown below:
```
I like eating apples. <start>This is the cancer pathology sentence.<end>How about playing LOL?
```
The pathology_alone is the label which is 'Yes' or 'No'

Here is a piece of example dataset extracted from train dataset

```
id | patient_id | note | pathology_alone |
123 | 12345 | <start>pathology sent<end>. abcde | Yes
456 | 45678 | this does not contain pathology. | No

```

### Train and Evaluate
The file main.py is the main entry to train and evaluate the model. Please read the code to know the parameters you can configure to train the model. Here are the most common parameters to tune when training a model
* data_dir:  the input directory where the training and testing dataset are saved.
* num_train_epochs: the training epochs
* max_sent_length: number of sentences you want to predict together
* max_seq_length: the length of sentence with the unit in token
* output_dir: the output directory where the model and performance results are saved. And the default output folder will be /outputs
* do_train: wheter to train a model
* do_eval: whether to run evaluation on the test set

This is an example to train and evaluate the model simultaneously.
```
cd SenTag/src
python main.py --do_train --do_eval --num_train_epoch 3 --data_dir ../data/ehr --train_batch_size 4 --max_seq_length 64 --max_sent_length 8
```

This is an example to evaluate the model only, if you already have trained a model, and evaluate with a new test datatest
```
cd SenTag/src
python main.py --do_eval --data_dir ../data/ehr --max_seq_length 64 --max_sent_length 8
```

After finishing the training and evaluation, the model outputs and performance results will be saved under the folder /outputs, if you use the default output directory.

## Model Deoplying (Inference) Instruction
- Download the latest model file from team folder 

- Copy the model file (pytorch_model.bin) to the folder SenTag_serving/outputs 

- Refer to run_inference.py to see the function will be applied, including inputs and outputs. 
    - Input: Any JSON objects. 
    - Output: A JSON object {index: {"encounterDate": date, "results": {"whole_text": text, "1": cancer sentence}}}

Here is an exmpale of output format
```
{
    "0": {
        "encounterDate": "5/28/2019 7:31:00 PM",
        "results": {
            "whole_text": "The first cancer pathology sentence. This is not pathology. This is the second cancer pathology sentence"
                        "1": [
                "This is the second cancer pathology sentence",
                "The first cancer pathology sentence."
            ]
        }
    }
}
```


