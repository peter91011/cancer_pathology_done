from collections import defaultdict
import torch
from utils import get_entities, remove_tags
import collections
import json
import os
from sklearn.metrics import classification_report
import pandas as pd
from config import DATA_DIR


def evaluate(args, model, tokenizer, dataloader, labels_list):
    print("Start to evaluate the model...")
    eval_loss = 0.0
    nb_eval_steps = 0
    all_predictions = defaultdict(dict)
    all_labels = defaultdict(dict)
    all_sentences_input_ids = defaultdict(dict)
    idx = 0

    lab = []
    pre = []
    logit0 = []
    logit1 = []
    mm = torch.nn.Softmax(dim=2)
    a = 0

    for batch in dataloader:
        model.eval()

        inputs = {k: v.to(args.device) for k, v in batch.items()}


        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            for j in mm(logits).cpu().numpy().reshape(mm(logits).shape[0] * mm(logits).shape[1], mm(logits).shape[2]):
                logit0.append(j[0])
                logit1.append(j[1])
            eval_loss += outputs['loss'].item()
            pred_labels = torch.flatten(torch.argmax(logits, dim=-1)).tolist()

        true_label_ids = torch.flatten(inputs['label_ids']).cpu().numpy().tolist()
        lab += true_label_ids
        pre += pred_labels
        a += 1
    print('current max_seq_len is '+str(batch['sentences_input_ids'].shape[2]))

    pp = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    pp['lab'] = lab[:len(pp)]
    pp['pre'] = pre[:len(pp)]
    pp['softmax0'] = logit0[:len(pp)]
    pp['softmax1'] = logit1[:len(pp)]

    report_dict = classification_report(lab[:len(pp)], pre[:len(pp)], output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # save the file
    df_report.to_csv(os.path.join(args.output_dir, "performance_results.csv"),index=False)
    pp.to_csv(os.path.join(args.output_dir, "evaluation_output.csv"),index=False)