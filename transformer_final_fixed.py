import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from metrics import evaluate_model, plot_metrics


#hyperparameters
batch_size = 256
block_size = 128  # Increased from 64 to 128
max_iters = 30000  # Increased from 15000 to 30000
eval_interval = 1500
learning_rate = 1e-3  # Decreased from 3e-3 to 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 128  # Increased from 32 to 128
dropout = 0.2
n_head = 6
n_layer = 6

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
#train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
  #generate small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y


@torch.no_grad()
def estimate_loss():
   out={}
   model.eval()
   for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
         X,Y = get_batch(split)
         logits,loss = model(X,Y)
         losses[k] = loss.item()
      out[split] = losses.mean()

   model.train()
   return out


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    # compute attention scores(affinities)
    wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # root(dk) to reduce variance
    
    # Handle sequences longer than block_size by creating a mask of the right size
    if T <= block_size:
        # Use pre-computed mask for efficiency if sequence is shorter than block_size
        mask = self.tril[:T, :T] == 0
    else:
        # Create a new mask dynamically if sequence is longer than block_size
        mask = torch.tril(torch.ones(T, T, device=device)) == 0
    
    # Apply mask
    wei = wei.masked_fill(mask, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    
    # perform aggregation of values
    v = self.value(x)
    out = wei @ v
    return out


#multiple attention in parallel
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(num_heads * head_size, n_embed)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out


#feedforward
class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout), #to prevent overfitting
    )
  
  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  '''Communication followed by computation'''
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class transformer(nn.Module):
  
  def __init__(self, vocab_size, n_embed):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    # Position embedding table dynamically resizes to match block_size
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) #final layer norm
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # idx and targets are both (B, T) tensor of integers
    token_embed = self.token_embedding_table(idx) # (B, T, n_embed)
    
    # Handle positional embeddings for sequences longer than block_size
    if T <= block_size:
        # Use pre-computed position embeddings for efficiency
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
    else:
        # For longer sequences, we need to handle differently
        # Use modulo to repeat positions (better for longer sequences)
        positions = torch.arange(T, device=device) % block_size
        pos_embed = self.position_embedding_table(positions)
    
    x = token_embed + pos_embed # (B, T, n_embed)
    x = self.blocks(x) # (B, T, n_embed)
    x = self.ln_f(x) # (B, T, n_embed)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      # Added label_smoothing=0.1 to the CrossEntropy loss
      loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens if it exceeds block_size
      idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sample index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = transformer(vocab_size, n_embed)
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
plt.savefig('transformer_loss_plot.png')
plt.close()

# Generate from the model
print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=1000)[0].tolist())

# Save the generated output to a file
with open("transformer_output.txt", "w") as file:
    file.write(output)

print("Sample of generated text:")
print(output[:500] + "...")

# Evaluate the transformer model - Using our evaluate_model function
print("Evaluating the model...")
try:
    transformer_metrics = evaluate_model(model, data, decode, device)
    print("Transformer Model Metrics:", transformer_metrics)

    # Save metrics to file
    with open("transformer_metrics.txt", "w") as f:
        for k, v in transformer_metrics.items():
            f.write(f"{k}: {v}\n")
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Skipping detailed evaluation due to error.")

print("Done! Output saved to transformer_output.txt")