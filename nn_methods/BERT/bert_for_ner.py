import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torchcrf import CRF

from transformers import (
    PreTrainedModel, 
    BertConfig,
    BertModel,
    add_start_docstrings
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable

# Code from sources of Hugginface Transofrmers

BertLayerNorm = torch.nn.LayerNorm

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = {}
    load_tf_weights = False
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# My models

@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """
)
class BertLstmCrfNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # Можно перенести дропаут в лстм
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, batch_first=True, bidirectional=True)
        self.position_wise_ff = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        outs, hc = self.bilstm(sequence_output)
        emissions = self.position_wise_ff(outs)
        # Skip [CLS] token
        ems = emissions[:, 1:, :]
        m = attention_mask[:, 1:].type(torch.bool)
        normed_labels = labels[:, 1:].clone()
        # Change padding label (-100 e.g.) to 0 label (needs for work of pytorch-crf layer)
        normed_labels[normed_labels < 0] = 0
        
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf.forward(ems, normed_labels, mask=m, reduction="mean"), self.crf.decode(ems, mask=m)

            tmp = []
            for seq in sequence_of_tags:
                tmp.append(np.pad(seq, (1, labels.data.size()[1] - len(seq) - 1), "constant", constant_values=(0, 0)))
            sequence_of_tags = np.array(tmp)

            # Inver NLL to best view
            return -log_likelihood, sequence_of_tags, outputs[2:] 
        else:
            tmp = []
            sequence_of_tags = self.crf.decode(ems, mask=m)
            for seq in sequence_of_tags:
                tmp.append(np.pad(seq, (1, labels.data.size()[1] - len(seq) - 1), "constant", constant_values=(0, 0)))
            sequence_of_tags = np.array(tmp)
            return secuence_of_tags, outputs[2:]


class BertCnnConfig(BertConfig):
    def __init__(
        self,
        cnn_filters=128,
        cnn_kernel_size=3,
        cnn_blocks=1,
        max_seq_len=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.cnn_filters = cnn_filters # == max_input_size
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_blocks = cnn_blocks
        self.max_seq_len = 128


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """
)
class BertCnnCrfNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cnn_blocks = []
        for i in range(config.cnn_blocks):
            block = nn.Sequential(
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size), 
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size), 
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size, dilation=2)              
            )
            self.cnn_blocks.append(block)
        self.cnn_blocks = nn.Sequential(*self.cnn_blocks)

        def conv1d_out_size(f, k, d):
            return (f - d * (k - 1))

        l_out = conv1d_out_size(config.hidden_size, config.cnn_kernel_size, 1)
        l_out = conv1d_out_size(l_out, config.cnn_kernel_size, 1)
        l_out = conv1d_out_size(l_out, config.cnn_kernel_size, 2)
        self.position_wise_ff = nn.Linear(l_out * config.cnn_blocks, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Skip [CLS] token
        #sequence_output = outputs[0][:, 1:]
        sequence_output = outputs[0]
        m = attention_mask[:, 1:].type(torch.bool)
        normed_labels = labels[:, 1:].clone()
        # Change padding label (-100 e.g.) to 0 label (needs for work of pytorch-crf layer)
        normed_labels[normed_labels < 0] = 0

        sequence_output = self.dropout(sequence_output)

        outs = []
        for block in self.cnn_blocks:
            outs.append(block(sequence_output))
        outs = torch.cat(outs, 2)  

        emissions = self.position_wise_ff(outs)
        ems = emissions[:, 1:]

        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf.forward(ems, normed_labels, mask=m, reduction="token_mean"), self.crf.decode(ems, mask=m)

            tmp = []
            for seq in sequence_of_tags:
                tmp.append(np.pad(seq, (1, labels.data.size()[1] - len(seq) - 1), "constant", constant_values=(0, 0)))
            sequence_of_tags = np.array(tmp)

            # Inver NLL to best view
            return -log_likelihood, sequence_of_tags, outputs[2:] 
        else:
            tmp = []
            sequence_of_tags = self.crf.decode(ems, mask=m)
            for seq in sequence_of_tags:
                tmp.append(np.pad(seq, (1, labels.data.size()[1] - len(seq) - 1), "constant", constant_values=(0, 0)))
            sequence_of_tags = np.array(tmp)
            return secuence_of_tags, outputs[2:]


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """
)
class BertLstmLinearNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # Можно перенести дропаут в лстм
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        outs, hc = self.bilstm(sequence_output)
        logits = self.classifier(outs)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """
)
class BertCnnLinearNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cnn_blocks = []
        for i in range(config.cnn_blocks):
            block = nn.Sequential(
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size), 
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size), 
                nn.Conv1d(config.cnn_filters, config.cnn_filters, kernel_size=config.cnn_kernel_size, dilation=2)              
            )
            self.cnn_blocks.append(block)
        self.cnn_blocks = nn.Sequential(*self.cnn_blocks)

        def conv1d_out_size(f, k, d):
            return (f - d * (k - 1))

        l_out = conv1d_out_size(config.hidden_size, config.cnn_kernel_size, 1)
        l_out = conv1d_out_size(l_out, config.cnn_kernel_size, 1)
        l_out = conv1d_out_size(l_out, config.cnn_kernel_size, 2)
        self.classifier = nn.Linear(l_out * config.cnn_blocks, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        outs = []
        for block in self.cnn_blocks:
            outs.append(block(sequence_output))
        outs = torch.cat(outs, 2)  

        logits = self.classifier(outs)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)





