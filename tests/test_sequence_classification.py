import torch
import torch.nn.functional as F
from transformers import RobertaConfig

from models.transformer import TransformerEncoder
from models.heads import (
    TransformerForSequenceClassification,
    MeanPooling,
)

# -----------------------
# Config
# -----------------------
max_seq_len = 128
num_labels = 5


def build_model():
    config = RobertaConfig(
        vocab_size=250,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=256 * 4,
        max_position_embeddings=max_seq_len + 2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.05,
    )

    # Custom flags
    config.use_rope = True
    config.attention_type = "sliced_relu"
    config.num_labels = num_labels

    encoder = TransformerEncoder(config)

    model = TransformerForSequenceClassification(
        encoder=encoder,
        config=config,
        num_labels=num_labels,
        pooling="cls",   # or "cls"
    )

    return model


# -----------------------
# Run test
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model().to(device)
model.train()

batch_size = 4
seq_len = 32

# Random token ids
input_ids = torch.randint(
    low=0,
    high=model.encoder.config.vocab_size,
    size=(batch_size, seq_len),
    device=device,
)

# Attention mask
attention_mask = torch.ones(
    batch_size,
    seq_len,
    device=device,
)

# Random labels
labels = torch.randint(
    low=0,
    high=num_labels,
    size=(batch_size,),
    device=device,
)

# Forward
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

print(outputs)

print("Forward pass OK")
print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())

# Backward
outputs.loss.backward()
print("Backward pass OK")