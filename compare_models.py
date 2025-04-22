from metrics import plot_metrics

def load_metrics(filename):
    metrics = {}
    try:
        with open(filename) as f:
            lines = f.readlines()
        
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)  # Split only on the first colon
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    metrics[key] = value
                except ValueError:
                    print(f"Warning: Could not parse value for {key} in {filename}")
    except FileNotFoundError:
        print(f"Warning: Metrics file {filename} not found")
    
    return metrics

# Try to load both metric files
bigram_metrics = load_metrics("bigram_metrics.txt")
transformer_metrics = load_metrics("transformer_metrics.txt")

# Print what was loaded to help debug
print("Bigram metrics loaded:", bigram_metrics)
print("Transformer metrics loaded:", transformer_metrics)

# Only include models with valid metrics
all_metrics = {}
if bigram_metrics:
    all_metrics["Bigram"] = bigram_metrics
if transformer_metrics:
    all_metrics["Transformer"] = transformer_metrics

if all_metrics:
    print("Plotting metrics comparison...")
    plot_metrics(all_metrics)
else:
    print("No valid metrics found to plot.")