import torch

from transformers import (
    AutoTokenizer,
)
from transformers.models.bert.modeling_bert import (
    BertConfig,
)
from transformers import WEIGHTS_NAME
import nltk
from torch.utils.data import DataLoader
from utils import *
from models.sentag import SenTag
from config import *
import collections
import json
import time
import sys
cur_time = time.strftime("%m%d%H%M%S", time.localtime())

class InferModel:
    def __init__(self, device, bert_model, label_list, model_dir, max_seq_length, max_sent_length, rule_features_path, batch_size):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.config = BertConfig.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length
        self.max_sent_length = max_sent_length

        self.model_dir = model_dir

        with open(rule_features_path) as r:
            self.rule_features_dict = {}
            lines = r.readlines()
            for i in range(len(lines)):
                self.rule_features_dict[i] = lines[i].replace('\n', '')

        self.num_rule_features = len(self.rule_features_dict)
        self.sentag_model = SenTag.from_pretrained(bert_model, config=self.config,
                                                   label_list=label_list, num_rule_features=self.num_rule_features, device=self.device)
        self.sentag_model.load_state_dict(torch.load(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu"))
        self.sentag_model.eval()
        self.sentag_model.to(self.device)
        self.batch_size = batch_size

    def clean_text(self, text):
        # remove html tags
        reg = re.compile('<.*?>')
        text = re.sub(reg, ' ', text)

        # remove unicode
        text = text.encode("ascii", "ignore").decode()

        # remove html 
        text = text.replace("&#x2022;", "")
        text = text.replace("&#xA0;", "")
        text = text.replace("&#xB7;", "")
        text = text.replace("&lt;", "")
        text = text.replace("&gt;", "")
        text = text.replace("&apos;", "")
        text = text.replace("&amp;", "")
        text = text.replace("&#xB0", "")

        # remove extra space 
        text = re.sub(' +', ' ', text)
        return text.strip()

    def pre_processing(self, text):
        # split in sentences
        if type(text) == list:
            sentences = text
        else:
            if '<br/>' in text:
                text = text.replace('<br/>', '[SEP]')
            if '<br />' in text:
                text = text.replace('<br />', '[SEP]')
            if '</paragraph>' in text:
                text = text.replace('</paragraph>', '[PARA]')

            if '[SEP]' not in text:
                 text = text.replace(',', '[SEP]')

            if '[PARA]' in text:
                sentences = text.split('[PARA]')
            else:
                sentences = [text]

        sentences_list = list()
        for para in sentences:
            para_text = self.clean_text(para)
            if len(para_text) == 0:
                continue
            
            sentences = para_text.split('[SEP]')
            
            for sent in sentences:
                sentences_list += nltk.sent_tokenize(sent)

        return sentences_list

    def convert_features(self, example):
        self.all_sents = sentences = self.pre_processing(example)
        sentences_tokens_new = [nltk.word_tokenize(sent.strip()) for sent in sentences]

        sentences_input_ids = list()
        sentences_input_mask = list()
        sentences_type_ids = list()
        rule_features = list()
        features = list()

        for sent in sentences_tokens_new:
            rfeat = [0] * self.num_rule_features

            for k in self.rule_features_dict:
                v = self.rule_features_dict[k].lower()
                rfeat[k] = 1 if v in [j.lower() for j in sent] else 0

            sent_feature = self.tokenizer(sent, is_split_into_words=True, max_length=self.max_seq_length,
                                          padding="max_length", truncation=True)

            sentences_input_ids.append(sent_feature['input_ids'])
            sentences_input_mask.append(sent_feature['attention_mask'])
            sentences_type_ids.append(sent_feature['token_type_ids'])
            rule_features.append(rfeat)

        if len(sentences_input_ids) % self.max_sent_length != 0:
            empty_sentence = self.tokenizer([], is_split_into_words=True, max_length=self.max_seq_length,
                                       padding="max_length", truncation=True)

            while len(sentences_input_ids) % self.max_sent_length != 0:
                sentences_input_ids.append(empty_sentence['input_ids'])
                sentences_input_mask.append(empty_sentence['attention_mask'])
                sentences_type_ids.append(empty_sentence['token_type_ids'])
                rule_features.append([0] * self.num_rule_features)

        for i in range(0, len(sentences_input_ids), self.max_sent_length):
            features.append({'sentences_input_ids': sentences_input_ids[i: i + self.max_sent_length],
                             'sentences_input_mask': sentences_input_mask[i: i + self.max_sent_length],
                             'sentences_type_ids': sentences_type_ids[i: i + self.max_sent_length],
                             'rule_features': rule_features[i: i + self.max_sent_length],
                             'sentences_input_len': self.max_sent_length})

        def collate_fn(batch):
            def convert_to_tensor(key):
                if isinstance(key, str):
                    tensors = [torch.tensor(o[1][key], dtype=torch.long) for o in batch]
                else:
                    tensors = [torch.tensor(o, dtype=torch.long) for o in key]

                return torch.stack(tensors)

            ret = dict(sentences_input_ids=convert_to_tensor('sentences_input_ids'),
                       sentences_input_mask=convert_to_tensor('sentences_input_mask'),
                       sentences_type_ids=convert_to_tensor('sentences_type_ids'),
                       rule_features=convert_to_tensor('rule_features'),
                       sentences_input_len=convert_to_tensor('sentences_input_len'))

            return ret

        dataloader = DataLoader(list(enumerate(features)), batch_size=self.batch_size, collate_fn=collate_fn)
        return dataloader

    def predict(self, example):

        dataloader = self.convert_features(example)
        mm = torch.nn.Softmax(dim=2)
        logit0 = []
        logit1 = []
        pre = []
        result_dic = {}
        result_dic['whole_text'] = ''
        result_dic[1] = []

        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.sentag_model(**inputs)
                logits = output['logits']
                pred_labels = torch.flatten(torch.argmax(logits, dim=-1)).tolist()

            sentences_input_ids = inputs['sentences_input_ids']

            pre += pred_labels

        for i in range(len(self.all_sents)):
            if pre[i] == 1:

                result_dic[1].append(self.all_sents[i])

        result_dic['whole_text'] = ' '.join(self.all_sents)


        return result_dic


model = InferModel(device=DEVICE, bert_model=BERT_MODEl,
                   label_list=LABEL_LIST, model_dir=MODEL_DIR,
                   max_seq_length=MAX_SEQ_LEN, max_sent_length=MAX_SENT_LEN,
                   rule_features_path=RULE_FEATURES_PATH,
                   batch_size=BATCH_SIZE)


def get_model():
    return model

