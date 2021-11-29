import torch
import os
from torch.utils.data import TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from data_processor import SenTagProcessor, InputFeatures
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import copy


def convert_examples_to_features(examples, label_list, tokenizer, max_seq_length, max_sent_length, rule_features_length):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (idx, example) in enumerate(examples):
        if idx % 1000 == 0:
            print("Converting examples to features: {} of {}".format(idx, len(examples)))

        sentences_input_ids = list()
        sentences_input_mask = list()
        sentences_type_ids = list()
        labels = example.labels[:max_sent_length]
        rule_features = example.rule_features[:max_sent_length]
        for sent in example.sentences[:max_sent_length]:
            sent_feature = tokenizer(sent, is_split_into_words=True, max_length=max_seq_length,
                                     padding="max_length", truncation=True)

            sentences_input_ids.append(sent_feature['input_ids'])
            sentences_input_mask.append(sent_feature['attention_mask'])
            sentences_type_ids.append(sent_feature['token_type_ids'])

        # if the sentences in this example are less than max_sent_length, then padding with empty sentence
        empty_sentence = tokenizer([], is_split_into_words=True, max_length=max_seq_length,
                                   padding="max_length", truncation=True)
        default_rule_feat = [0] * rule_features_length
        while len(sentences_input_ids) < max_sent_length:
            sentences_input_ids.append(empty_sentence['input_ids'])
            sentences_input_mask.append(empty_sentence['attention_mask'])
            sentences_type_ids.append(empty_sentence['token_type_ids'])
            rule_features.append(default_rule_feat)
            labels.append(0)

        label_ids = [label_map[label] for label in labels]

        assert len(sentences_input_ids) == max_sent_length
        assert len(sentences_input_mask) == max_sent_length
        assert len(sentences_type_ids) == max_sent_length
        assert len(label_ids) == max_sent_length
        assert len(rule_features) == max_sent_length

        features.append(InputFeatures(sentences_input_ids=sentences_input_ids,
                                      sentences_input_mask=sentences_input_mask,
                                      sentences_type_ids=sentences_type_ids,
                                      sentences_input_len=max_sent_length,
                                      rule_features=rule_features,
                                      label_ids=label_ids))

    return features


def load_examples(args, tokenizer, data_type):
    if args.local_rank not in (-1, 0) and data_type == "train":
        torch.distributed.barrier()

    processor = SenTagProcessor()
    processor.get_rule_features(args.data_dir)
    processor.get_config_file()

    if data_type == 'train' and args.debug:
        examples = processor.get_debug_examples(args.data_dir)
    elif data_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif data_type == 'test' and args.debug:
        examples = processor.get_debug_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    label_list = processor.get_labels()
    
    print("Creating features from the dataset...")
    rule_features_len = len(processor.rule_features_dict)
    features = convert_examples_to_features(examples, label_list, tokenizer, args.max_seq_length, args.max_sent_length, rule_features_len)

    if args.local_rank == 0 and data_type == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def convert_to_tensor(key):
            if isinstance(key, str):
                tensors = [torch.tensor(getattr(o[1], key), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in key]

            return torch.stack(tensors)

        ret = dict(sentences_input_ids=convert_to_tensor('sentences_input_ids'),
                   sentences_input_mask=convert_to_tensor('sentences_input_mask'),
                   sentences_type_ids=convert_to_tensor('sentences_type_ids'),
                   sentences_input_len=convert_to_tensor('sentences_input_len'),
                   rule_features=convert_to_tensor('rule_features'),
                   label_ids=convert_to_tensor('label_ids'))

        return ret

    if data_type == "train":
        sampler = RandomSampler(features) if args.local_rank == -1 else DistributedSampler(features)
        dataloader = DataLoader(list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size,
                                collate_fn=collate_fn)
    else:
        print('here creating eval dataloader')
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, label_list, rule_features_len
