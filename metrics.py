import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

def calculate_bleu(reference, candidate, max_n=4):
    weights = tuple((1. / max_n for _ in range(max_n)))
    return sentence_bleu([reference], candidate, weights=weights)

@torch.no_grad()
def evaluate_model(model, data, decode, device, num_tokens=500):
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=num_tokens)[0]
    generated_text = decode(generated.tolist())
    reference_text = decode(data[:num_tokens].tolist())

    # Metrics
    bleu = calculate_bleu(list(reference_text), list(generated_text))

    inputs = torch.tensor(data[:num_tokens-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(data[1:num_tokens], dtype=torch.long, device=device).unsqueeze(0)
    logits, loss = model(inputs, targets)
    cross_entropy = loss.item()
    perplexity = torch.exp(loss).item()

    return {
        'BLEU': bleu,
        'CrossEntropy': cross_entropy,
        'Perplexity': perplexity
    }

def plot_metrics(metrics_dict, title="Language Model Comparison"):
    labels = list(next(iter(metrics_dict.values())).keys())
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))

    for i, (model_name, scores) in enumerate(metrics_dict.items()):
        values = list(scores.values())
        offset = (i - len(metrics_dict) / 2) * width + width / 2
        plt.bar([j + offset for j in x], values, width=width, label=model_name)

    plt.xticks(ticks=x, labels=labels)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
