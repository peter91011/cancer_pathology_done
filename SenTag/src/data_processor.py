import json
import copy
import os
import nltk
import re
import pandas as pd
from config import RULE_FEATURES_PATH

class InputFeatures(object):
    def __init__(self, sentences_input_ids, sentences_input_mask,
                 sentences_type_ids, sentences_input_len, rule_features, label_ids):
        self.sentences_input_ids = sentences_input_ids
        self.sentences_input_mask = sentences_input_mask
        self.sentences_type_ids = sentences_type_ids
        self.label_ids = label_ids
        self.rule_features = rule_features
        self.sentences_input_len = sentences_input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExample(object):
    def __init__(self, guid, sentences, rule_features, labels):
        self.guid = guid
        self.sentences = sentences
        self.rule_features = rule_features
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SenTagProcessor(object):
    def __init__(self):
        self.rule_features_dict = None

    def get_debug_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "debug.csv")), 'train')

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.csv")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.csv")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.csv")), 'test')

    def get_labels(self):
        return [0,1]

    def _word_tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def get_config_file(self):
        from config import MAX_SENT_LEN
        self.max_sent_length = MAX_SENT_LEN

    def get_rule_features(self, data_dir):

        data_path = RULE_FEATURES_PATH
        with open(data_path) as r:
            self.rule_features_dict = {}
            lines = r.readlines()
            for i in range(len(lines)):
                self.rule_features_dict[i] = lines[i].replace('\n', '')

    def _read_data(self, input_file):
        print('max sent length: ',self.max_sent_length)
        data = []
        df = pd.read_csv(input_file)
        sentences = [nltk.word_tokenize(i.strip()) for i in df['note']]
        labels = df['label'].to_list()

        add_sents = []
        add_labels = []
        rule_features = []
        cnt = 0
        for i in range(len(sentences)):

            rfeat = [0] * len(self.rule_features_dict)
            for k in self.rule_features_dict:
                v = self.rule_features_dict[k].lower()
                rfeat[k] = 1 if v in [j.lower() for j in sentences[i]] else 0
            rule_features.append(rfeat)
            add_sents.append(sentences[i])
            add_labels.append(labels[i])
            cnt += 1
            if cnt % self.max_sent_length == 0:
                data.append({"sentences": add_sents, "rule_features": rule_features, "labels": add_labels})
                add_sents = []
                add_labels = []
                rule_features = []
            if cnt % 10000 == 0:
                print('preparing examples for count ' + str(cnt))

        if len(add_sents) != 0:
            data.append({"sentences": add_sents, "rule_features": rule_features, "labels": add_labels})

        return data

    def _create_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = "%s-%s" % (data_type, i)
            sentences = line['sentences']
            rule_features = line['rule_features']
            labels = line['labels']
            examples.append(InputExample(guid=guid, sentences=sentences, rule_features=rule_features, labels=labels))

        return examples




