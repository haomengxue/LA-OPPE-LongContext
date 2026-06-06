# Checkpoints

This directory may contain only lightweight LA-OPPE delta weights.

The repository does NOT include:
- Llama 3.2-1B base model weights
- full model checkpoints
- Hugging Face cache files
- third-party dataset files

If `laoppe_l2_th2048_delta.pt` is provided, it should contain only newly introduced LA-OPPE parameters such as:
- `laoppe_bias.feature_weight`
- `laoppe_bias.alpha_raw`

Users need to download the base model separately according to its license.
