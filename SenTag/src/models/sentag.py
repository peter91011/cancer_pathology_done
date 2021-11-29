import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertConfig,
    BertPreTrainedModel,
)
from .layers import SelfAttention

class SenTag(BertPreTrainedModel):
    def __init__(self, config, label_list, num_rule_features, device):
        super(SenTag, self).__init__(config)
        print('device here is ',device)
        # get bert model
        self.bert = BertModel(config)
        self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.label_list = label_list
        self.linear = nn.Linear(self.config.hidden_size*2, self.config.hidden_size)
        self.sentence_lstm = nn.LSTM(input_size=self.config.hidden_size,
                                     hidden_size=self.config.hidden_size // 2,
                                     batch_first=True,
                                     bidirectional=True)

        self.self_attention = SelfAttention(hidden_size=self.config.hidden_size, dim=self.config.hidden_size)
        self.ln = LayerNorm(self.config.hidden_size*2)
        self.classifier = nn.Linear(num_rule_features + self.config.hidden_size, len(self.label_list))
        label2id = {k: i for i, k in enumerate(self.label_list)}
        # self.crf = CRF(num_tags=len(self.label_list))


        self.init_weights()

    def forward(self, sentences_input_ids=None, sentences_input_mask=None,
                sentences_type_ids=None, sentences_input_len=None, rule_features=None, label_ids=None):

        batch_size = sentences_input_ids.shape[0]
        seq_len = sentences_input_ids.shape[2]
        sentences_feature = list()

        for i in range(batch_size):

            bert_output = self.bert(sentences_input_ids[i], sentences_type_ids[i], sentences_input_mask[i])

            last_hidden_state = bert_output['last_hidden_state']
            pooler_output = bert_output['pooler_output']

            # we define the sentence features as the average the sequence output and concatenate with pooler output
            # mask the padding tokens for each sentence
            sentence_feature = torch.unsqueeze(sentences_input_mask[i], 2) * last_hidden_state

            # perform self attention layer to re-estimate sentence features
            sentence_feature, attention_weights = self.self_attention(sentence_feature, sentence_feature,
                                                                      sentence_feature)
            # lstm
            sentence_feature, _ = self.sentence_lstm(sentence_feature)

            # compute the average pooling the sentence feature
            sentence_feature = torch.sum(sentence_feature, dim=1) / seq_len
            sentence_feature = torch.cat((sentence_feature, pooler_output), dim=1)
            sentences_feature.append(sentence_feature)

        sentences_feature = torch.stack(sentences_feature)

        # passing sentence feature to layer normalization
        sentences_feature = self.ln(sentences_feature)
        sentences_feature = self.linear(sentences_feature)
        sentences_feature = self.dropout(sentences_feature)

        # concatenate sentence features with rule-based features
        outputs = torch.cat((sentences_feature, rule_features), dim=2)
        logits = self.classifier(outputs)
        loss = None

        if label_ids is not None:
            cro_loss = nn.CrossEntropyLoss(weight=torch.tensor([ 0.5027, 92.2370]).to(self.device))
            # cro_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5027, 180.2370]).to(self.device))
            loss = cro_loss(logits.reshape(logits.shape[0] * logits.shape[1], len(self.label_list)),
                            torch.flatten(label_ids))
        return {'loss': loss, 'logits': logits}