## Pre-trained Weights

### Model Configurations

We provide multiple model scales of **DARE-EEG**, covering lightweight to large-capacity architectures.
The detailed configurations and parameter sizes are summarized below.

| Model | Params (M) | Embed Dim | Embed Num | Depth        | Num Heads |
|------:|-----------:|----------:|----------:|--------------|-----------|
| Nano  | 0.6        | 64        | 1         | [2, 2, 4]    | 4         |
| Light | 3.8        | 128       | 1         | [6, 6, 6]    | 4         |
| Small | 14.7       | 256       | 1         | [6, 6, 6]    | 4         |
| Base  | 19.9       | 256       | 4         | [8, 8, 8]    | 4         |
| Deep  | 77.8       | 512       | 4         | [8, 8, 8]    | 8         |

**Notes:**
- *Embed Dim* denotes the embedding dimension of each token.
- *Embed Num* indicates the number of parallel embedding streams.
- *Depth* specifies the number of blocks in each stage.
- *Num Heads* is the number of attention heads per block.

In this repository, due to GitHub's file size limitations, we can only provide the **Nano** model (file size less than 25MB).
To prevent identity leaks, more Pre-trained model weights are not released during the double-blind review process. But the weights will be shared via external storage (e.g., cloud drive) after the anonymization period ends.

