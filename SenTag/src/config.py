import os
import torch

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT_PATH, "../outputs")
DATA_DIR = os.path.join(ROOT_PATH, "../data/ehr")
TRAIN_DATA = os.path.join(ROOT_PATH, "../data/ehr/cancer.csv")
BERT_MODEl = 'emilyalsentzer/Bio_ClinicalBERT'
PORT_NUMBER = 8005
MAX_SEQ_LEN = 64
MAX_SENT_LEN = 8
BATCH_SIZE = 4
RULE_FEATURES_PATH = os.path.join(ROOT_PATH, "../data/ehr/keyword.txt")
LABEL_LIST = [0, 1]
DEVICE = torch.device("cpu")
EPOCH = 1