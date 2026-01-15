import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class MeanPooling(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return hidden_states.mean(dim=1)


class CLSPooling(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states[:, 0] 
    

    

class TransformerForSequenceClassification(nn.Module):
    """
    Encoder + pooling + MLP classification head.
    HF-style output: SequenceClassifierOutput(loss, logits, ...)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config,
        num_labels: int,
        pooling: str = "mean",   # "mean" or "cls"
    ):
        super().__init__()
        self.encoder = encoder
        self.num_labels = int(num_labels)

        if pooling == "mean":
            self.pooler = MeanPooling()
        elif pooling == "cls":
            self.pooler = CLSPooling()
        else:
            raise ValueError(f"Unknown pooling={pooling!r}. Use 'mean' or 'cls'.")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        mlp_hidden = 4 * config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(mlp_hidden, self.num_labels),
        )

        # optional: store for convenience
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Accept both HF outputs and raw tensors
        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs

        pooled = self.pooler(last_hidden, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None) if hasattr(outputs, "hidden_states") else None,
            attentions=getattr(outputs, "attentions", None) if hasattr(outputs, "attentions") else None,
        )

            

