import argparse
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from typing import Dict, Optional, Tuple

# Import custom modules
from hrm_wrapper import BoxTensor
from hrm_wrapper import CenterSigmoidBoxTensor
from modules import BoxDecoder
from modules import HighwayNetwork
from modules import LinearProjection
from modules import SimpleDecoder
from constant import load_vocab_dict_hierachy
from constant import load_vocab_dict
from constant import TYPE_FILES


TRANSFORMER_MODELS = {
    "bert-large-uncased": (BertModel, BertTokenizer),
    "bert-large-uncased-whole-word-masking": (BertModel, BertTokenizer)
}


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func_l1 = nn.BCEWithLogitsLoss()
        self.sigmoid_fn = nn.Sigmoid()

    def define_loss(self,
                    logits_l1: torch.Tensor,
                    targets_l1: torch.Tensor,
                    logits_l2: torch.Tensor,
                    targets_l2: torch.Tensor,
                    weight = None) -> torch.Tensor:
        loss_l1 = self.loss_func_l1(logits_l1,targets_l1)
        loss_l2 = self.loss_func(logits_l2, targets_l2)
        loss = loss_l2 * 0.6 + loss_l1 * 0.4
        return loss

    def hypervol_indictor(self,
                    logits_l1: torch.Tensor,
                    targets_l1: torch.Tensor,
                    logits_l2: torch.Tensor,
                    targets_l2: torch.Tensor,
                    weight = None) -> torch.Tensor:
        z = 0.5
        loss_l1 = self.loss_func_l1(logits_l1, targets_l1)
        loss_l2 = self.loss_func(logits_l2, targets_l2)
        h_z = torch.log(torch.sum(loss_l1) + torch.sum(loss_l2))
        return h_z
    def forward(self, feed_dict: Dict[str, torch.Tensor]):
        pass


class TransformerVecModel(ModelBase):
    def __init__(self, args: argparse.Namespace, answer_num_l1: int, answer_num_l2: int):
        print("args.model_type:",args.model_type)
        super(TransformerVecModel, self).__init__()
        print("Initializing <{}> model...".format(args.model_type))
        _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
        self.transformer_tokenizer = _tokenizer_class.from_pretrained(
            args.bert_large_path)
        self.transformer_config = AutoConfig.from_pretrained(args.bert_large_path)
        self.encoder = _model_class.from_pretrained(args.bert_large_path)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.avg_pooling = args.avg_pooling
        self.reduced_type_emb_dim = args.reduced_type_emb_dim
        self.n_negatives = args.n_negatives
        output_dim = self.transformer_config.hidden_size
        self.transformer_hidden_size = self.transformer_config.hidden_size
        self.encoder_layer_ids = args.encoder_layer_ids
        if self.encoder_layer_ids:
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.randn(1), requires_grad=True)
                 for _ in self.encoder_layer_ids])

        if args.reduced_type_emb_dim > 0:
            output_dim = args.reduced_type_emb_dim
            self.proj_layer = HighwayNetwork(self.transformer_hidden_size,
                                             output_dim,
                                             2,
                                             activation=nn.ReLU())
        self.activation = nn.ReLU()
        self.classifier = SimpleDecoder(output_dim, answer_num_l1, answer_num_l2)

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            targets: Optional[torch.Tensor] = None,
            targets_l1: Optional[torch.Tensor] = None,
            targets_l2: Optional[torch.Tensor] = None,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=True if self.encoder_layer_ids else False)

        if self.avg_pooling:  # Averaging all hidden states
            outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(
                1) / inputs["attention_mask"].sum(1).unsqueeze(-1)
        else:  # Use [CLS]
            if self.encoder_layer_ids:
                _outputs = torch.zeros_like(outputs[0][:, 0, :])
                for i, layer_idx in enumerate(self.encoder_layer_ids):
                    _outputs += self.layer_weights[i] * outputs[2][layer_idx][:,
                                                        0, :]
                outputs = _outputs
            else:
                outputs = outputs[0][:, 0, :]

        outputs = self.dropout(outputs)

        if self.reduced_type_emb_dim > 0:
            outputs = self.proj_layer(outputs)

        logits = self.classifier(outputs)

        if targets is not None:
          loss = self.define_loss(logits, targets, targets_l1)
        else:
            loss = None
        return loss, logits



class TransformerBoxModel(TransformerVecModel):
    box_types = {
        "BoxTensor": BoxTensor,
        "CenterSigmoidBoxTensor": CenterSigmoidBoxTensor
    }

    def __init__(self, args: argparse.Namespace, answer_num_l1: int, answer_num_l2: int):
        super(TransformerBoxModel, self).__init__(args, answer_num_l1, answer_num_l2)
        self.goal = args.goal
        self.mc_box_type = args.mc_box_type
        self.type_box_type = args.type_box_type
        self.box_offset = args.box_offset
        self.inv_softplus_temp = args.inv_softplus_temp
        self.softplus_scale = args.softplus_scale
        self.word2id = load_vocab_dict(TYPE_FILES[args.goal])
        self.word2id_l1, self.word2id_l2 = load_vocab_dict_hierachy(TYPE_FILES[args.goal])
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2word_l1 = {v: k for k, v in self.word2id_l1.items()}
        self.id2word_l2 = {v: k for k, v in self.word2id_l2.items()}
        self.offset_scale_l1 = torch.randn(len(self.word2id_l1.keys()), 1)
        self.offset_scale_l2 = torch.randn(len(self.word2id_l2.keys()), 1)
        self.mention_offset = torch.tensor(2)
        self.args = args


        try:
            self.mc_box = self.box_types[args.mc_box_type]
        except KeyError as ke:
            raise ValueError(
                "Invalid box type {}".format(args.box_type)) from ke

        self.proj_layer = HighwayNetwork(
            self.transformer_hidden_size,
            args.box_dim * 2,
            args.n_proj_layer,
            activation=nn.ReLU())

        self.classifier = BoxDecoder(args,
                                     answer_num_l1,
                                     answer_num_l2,
                                     args.box_dim,
                                     args.type_box_type,
                                     args.per_gpu_train_batch_size,
                                     goal_data=args.goal,
                                     inv_softplus_temp=args.inv_softplus_temp,
                                     softplus_scale=args.softplus_scale,
                                     n_negatives=args.n_negatives,
                                     neg_temp=args.neg_temp,
                                     box_offset=args.box_offset,
                                     pretrained_box=None,
                                     use_gumbel_baysian=args.use_gumbel_baysian,
                                     gumbel_beta=args.gumbel_beta,
                                     offset_scale_l1=self.offset_scale_l1,
                                     offset_scale_l2 = self.offset_scale_l2
                                     )

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            targets: Optional[torch.Tensor] = None,
            targets_l1: Optional[torch.Tensor] = None,
            targets_l2: Optional[torch.Tensor] = None,
            batch_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_context_rep = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=True if self.encoder_layer_ids else False)

        # calculate offset scale
        tmp_scale_offset_l1 = torch.sigmoid(self.offset_scale_l1.repeat(1, self.args.box_dim))
        tmp_scale_offset_l2 = torch.sigmoid(self.offset_scale_l2.repeat(1, self.args.box_dim))


        # CLS
        if self.encoder_layer_ids:
            # Weighted sum of CLS from different layers
            _mention_context_rep = torch.zeros_like(
                mention_context_rep[0][:, 0, :])
            for i, layer_idx in enumerate(self.encoder_layer_ids):
                _mention_context_rep += self.layer_weights[i] * \
                                        mention_context_rep[2][layer_idx][:, 0, :]
            mention_context_rep = _mention_context_rep
        else:
            # CLS from the last layer
            mention_context_rep = mention_context_rep[0][:, 0, :]

        # Convert to box
        mention_context_rep = self.proj_layer(mention_context_rep)
        mention_offset = self.mention_offset
        mention_offset = torch.sigmoid(mention_offset)
        if self.mc_box_type == 'ConstantBoxTensor':

            mention_context_rep = self.mc_box.from_split(mention_context_rep,
                                                         self.box_offset)
        else:
            # 传入无用参数
            mention_context_rep = self.mc_box.from_split(mention_context_rep,mention_offset)

        # Compute probs (0-1 scale)
        if self.training and targets is not None:
            log_probs_l1, loss_weights, targets_l1,  targets_l2, log_probs_l2 = self.classifier(
                mention_context_rep,
                targets_l1=targets_l1,
                targets_l2=targets_l2,
                is_training=self.training,
                batch_num=batch_num,
            offset_scale_l1 = tmp_scale_offset_l1,
            offset_scale_l2 = tmp_scale_offset_l2)

        else:  # eval
            log_probs_l1, loss_weights, _, _l2, log_probs_l2 = self.classifier(mention_context_rep,
                                                         targets_l1=targets_l1,
                                                         targets_l2=targets_l2,
                                                         is_training=self.training,
                                                         batch_num=batch_num,
                                                         offset_scale_l1=tmp_scale_offset_l1,
                                                         offset_scale_l2 =tmp_scale_offset_l2)

        if targets is not None:
            loss = self.hypervol_indictor(log_probs_l1, targets_l1, log_probs_l2, targets_l2,weight=loss_weights)
        else:
            loss = None

        return loss, torch.exp(log_probs_l1), torch.exp(log_probs_l2)
