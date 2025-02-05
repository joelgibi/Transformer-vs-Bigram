##Tiny Shakespeare

This repository showcases the implementation and comparison of two language models: a Bigram Language Model and a Decoder-Only Transformer. Both implementations are contained in separate Python files for simplicity, with bigram.py focusing on a probabilistic approach that predicts the next character based on the previous one, and transformer.py implementing a modern self-attention-based architecture to model long-range dependencies effectively. The models are trained and evaluated using a character-level tokenized dataset of Shakespeare's complete works (input.txt). To set up, clone the repository, and ensure the dataset is available in the data/ folder. Training can be performed by running python bigram.py for the Bigram Language Model and python transformer.py for the Decoder-Only Transformer. Results demonstrate that while the Bigram Model is lightweight and fast, it struggles with long-range dependencies, whereas the Transformer generates more coherent and contextually relevant text. Future enhancements could include hyperparameter tuning, experimenting with alternative tokenization approaches, or visualizing the Transformer’s attention weights. 

## Installation & Dependencies

Ensure you have the following libraries installed before running the program:

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

### Install Required Packages:

Use the following command to install all dependencies:

pip install torch torchvision torchaudio matplotlib

---

### Usage

Once the dependencies are installed, you can run the script.
