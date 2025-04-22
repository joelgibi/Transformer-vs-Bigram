import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from metrics import evaluate_model, plot_metrics


# Hyperparameters
batch_size = 256  # Increased to match transformer
block_size = 128  # Increased from 8 to 128
max_iters = 30000  # Increased from 15000 to 30000
eval_interval = 1500
learning_rate = 1e-3  # Decreased to 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 128  # Added to match transformer though not directly used in bigram

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# Train-test split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

# Simple bigram model with slight enhancements
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (Batch, Time, Channel=vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Added label_smoothing=0.1 to CrossEntropy
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Train the model
print("Training the model...")
train_losses = []
val_losses = []

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot([i * eval_interval for i in range(len(train_losses))], train_losses, label='Train Loss')
plt.plot([i * eval_interval for i in range(len(val_losses))], val_losses, label='Val Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('bigram_loss_plot.png')
plt.close()

# Generate from model
print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=1000)[0].tolist())

with open("bigram_output.txt", "w") as file:
    file.write(output)

print("Sample of generated text:")
print(output[:500] + "...")

# Evaluate the model
print("Evaluating the model...")
try:
    bigram_metrics = evaluate_model(m, data, decode, device)
    print("Bigram Model Metrics:", bigram_metrics)

    # Save metrics with correct format
    with open("bigram_metrics.txt", "w") as f:
        for k, v in bigram_metrics.items():
            f.write(f"{k}: {v}\n")
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Skipping detailed evaluation due to error.")

print("Done! Metrics saved to bigram_metrics.txt and output saved to bigram_output.txt")