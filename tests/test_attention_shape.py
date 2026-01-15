import torch
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaPreTrainedModel,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

from models.transformer import *

max_seq_len = 128

def build_model():
    """
    Builds the MLM model (encoder + head). Keeps same signature expected by run_experiment.py.
    """
    config = RobertaConfig(
        vocab_size=250,          # roberta-base vocab
        hidden_size=256,
        num_attention_heads=8,    # hidden_size needs to be divisible by num_attention_heads
        num_hidden_layers=4,
        intermediate_size=256*4,
        max_position_embeddings=max_seq_len+2,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.05
    )
    # Custom flags 
    config.use_rope = True

    config.attention_type = "sliced_relu_bump"

    model = TransformerEncoder(config)

    return model


model = build_model()
print(model)

###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

batch_size = 4
seq_len = 32

# Random token ids in vocab range
input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(batch_size, seq_len),
    device=device
)

# Attention mask (all tokens visible)
attention_mask = torch.ones(
    batch_size,
    seq_len,
    device=device
)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
)

hidden = outputs.last_hidden_state


print("Forward pass OK")
print("Output hidden state shape:", hidden.shape)

# Optional: backward pass to test gradients
loss = hidden.mean()
loss.backward()

print("Backward pass OK")