import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

#hyperparameters
batch_size = 256
block_size = 64
max_iters = 15000
eval_interval = 1500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
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


#multiple attention in parallel
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
    self.proj = nn.Linear(num_heads * head_size, n_embed)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
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
  
  def forward(self,x):
    return self.net(x)


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    #compute attention scores(affinities)
    wei = q @ k.transpose(-2,-1) * C ** -0.5 #root(dk) to reduce variance
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    #perform aggregation of values
    v = self.value(x)
    out = wei @ v
    return out


class Block(nn.Module):
  '''Communication followed by computation'''
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)


  def forward(self,x):
    x = x +self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

#simple bigram model
#has no attention, the predicted piece is only dependent ont he previous piece

class transformer(nn.Module):
  
  def __init__(self, vocab_size, n_embed):
    super().__init__()
    #each token directly reads off the logits for the next token from a loookup table
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) #final layer norm

    self.lm_head = nn.Linear(n_embed, vocab_size)
    self.ffwd = FeedForward(n_embed)
    self.sa_heads = MultiHeadAttention(4, n_embed//4) #i.e 4 heads of 8-dimensional self-attention

    

  def forward(self, idx, targets=None):

    B,T = idx.shape
    # idx and targets are both (B, T) tensor of integers
    token_embed = self.token_embedding_table(idx) #(Batch=4,Time=8,n_embed)
    pos_embed = self.position_embedding_table(torch.arange(T, device=device))
    x = token_embed + pos_embed #x now holds the positional embedding too 
    x = self.blocks(x)
    logits = self.lm_head(x)#(Batch=4,Time=8,Channel=65)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)#cross entropy wants B,C,T dimension, hence reshape it
    return logits, loss

  def generate(self, idx, max_new_tokens):

    #idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      #crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      #get the predictions
      logits,loss = self(idx_cond)
      #focus only on the last time step
      logits = logits[:, -1,:] #becomes (B,C)
      #apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) #(B,C)
      #sample from the distribution
      idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
      #append sample index to the running sequence
      idx= torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx
  
model = transformer(vocab_size, n_embed)
m = model.to(device)

#create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=100000)[0].tolist())

# Save the generated output to a file
with open("transformer_output.txt", "w") as file:
    file.write(output)

print(output)

# Evaluate the transformer model
reference_text = list(text[:10000])  # Use a suitable reference text for evaluation
transformer_metrics = evaluate_model(m, reference_text)
print("Transformer Model Metrics:", transformer_metrics)
