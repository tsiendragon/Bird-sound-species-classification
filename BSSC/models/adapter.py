import math
from typing import Optional

import timm
import torch
import torch.nn as nn

from BSSC.utils import register_model

_HIDDEN_STATES_START_POSITION = 2


class TDNNLayer(nn.Module):
    def __init__(
        self,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        layer_id=0,
    ):
        super().__init__()
        self.in_conv_dim = (
            tdnn_dim[layer_id - 1] if layer_id > 0 else tdnn_dim[layer_id]
        )
        self.out_conv_dim = tdnn_dim[layer_id]
        self.kernel_size = tdnn_kernel[layer_id]
        self.dilation = tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavLMForXVector(nn.Module):
    def __init__(
        self,
        use_weighted_layer_sum=False,
        num_hidden_layers=24,
        hidden_size=1024,
        tdnn_dim=(512, 512, 512, 512, 512),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        feature_dim=256,
        **kwargs,
    ):
        super().__init__()

        self.wavlm = timm.create_model("wavlm_large")
        num_layers = num_hidden_layers + 1
        # transformer layers + input embeddings
        self.use_weighted_layer_sum = use_weighted_layer_sum
        if use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(hidden_size, tdnn_dim[0])

        tdnn_layers = [
            TDNNLayer(tdnn_dim, tdnn_kernel, tdnn_dilation, i)
            for i in range(len(tdnn_dim))
        ]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.features = nn.Sequential(
            nn.Linear(tdnn_dim[-1] * 2, feature_dim),
            nn.BatchNorm1d(feature_dim, eps=1e-05),
        )
        nn.init.constant_(self.features[1].weight, 1.0),
        self.features[1].weight.requires_grad = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init

        if isinstance(module, nn.Linear):
            # init linear layer with xavier_uniform
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
                nn.init.uniform_(module.bias, a=-k, b=k)

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_hidden_states = (
            True if self.use_weighted_layer_sum else output_hidden_states
        )
        #         source: torch.Tensor,
        # padding_mask: Optional[torch.Tensor] = None,
        # mask: bool = False,
        # ret_conv: bool = False,
        # output_layer: Optional[int] = None,
        # ret_layer_results: bool = False,

        outputs = self.wavlm(
            input_values,
            padding_mask=attention_mask,
            mask=False,
            output_layer=self.wavlm.cfg.encoder_layers,
            ret_layer_results=True,
        )

        if self.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling

        mean_features = hidden_states.mean(dim=1)
        std_features = hidden_states.std(dim=1)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.features(statistic_pooling)
        return output_embeddings


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40, feature_dim=256, num_classes=2, **kwargs):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.randn(feature_dim, num_classes), requires_grad=True
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # calculate similarity
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        logits = nn.functional.normalize(logits, dim=1)
        logits = torch.mm(logits, weight)

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        loss = self.loss(logits, labels)
        return loss


class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_classes, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_classes
        self.weight = nn.Parameter(
            torch.randn(input_dim, num_classes), requires_grad=True
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


@register_model
def wavlm_large_vector_v1(**kwargs):
    model = WavLMForXVector(
        use_weighted_layer_sum=False,
        num_hidden_layers=24,
        hidden_size=1024,
        tdnn_dim=(512, 512, 512, 512, 512),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        feature_dim=256,
    )
    return model


@register_model
def cosface_loss(**kwargs):
    model = CosFace(**kwargs)
    return model


if __name__ == "__main__":
    net = timm.create_model("wavlm_large_vector_v1")
    loss = timm.create_model("cosface_loss", feature_dim=256, num_classes=5)
    x = torch.randn(2, 16000)
    labels = torch.randint(0, 5, (2,))
    # test the codes
    print(net)
    print(loss)
    print(net(x).shape)
    print(loss(net(x), labels))
