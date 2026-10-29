# drumloop-models

Model files downloaded on first run by the Drumloop Android app.

## drums-v1.onnx

The drums specialist of **Demucs v4 `htdemucs_ft`** (Hybrid Transformer Demucs, fine-tuned),
by Alexandre Défossez / Meta AI — https://github.com/facebookresearch/demucs — released under the
MIT License (see LICENSE).

Changes: exported to ONNX (opset 17) with the STFT/iSTFT moved outside the graph (the app computes
them), only the drums output kept, and weights stored as float16 with Cast nodes (computation runs
in float32). Output matches PyTorch Demucs at ~80 dB SNR.

Inputs: `mix` [1,2,343980], `spec` [1,4,2048,336]. Outputs: `zx` [1,4,2048,336], `xt` [1,2,343980].

## beats-v1.onnx

**Beat This!** (`small0` checkpoint) beat and downbeat tracker by Foscarin, Schlüter & Widmer
(CPJKU, ISMIR 2024) — https://github.com/CPJKU/beat_this — MIT License, Copyright (c) 2024
Institute of Computational Perception, JKU Linz. Exported to ONNX (input `spect` [1,1500,128] log-mel,
outputs `beat`/`downbeat` logits [1,1500]) with float16 weight storage.
