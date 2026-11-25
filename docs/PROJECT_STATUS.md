# Project status

This snapshot includes:
- the UCF101 dataset loader for the processed ZIP layout;
- official split-1 lists and the audited 100-class train/validation/test split;
- RGB and optical-flow ResNet-18 backbones;
- bidirectional GRU temporal encoders with temporal attention;
- learnable two-stream fusion and a 100-class classifier;
- dataloader checks and a one-batch forward/backward/optimizer smoke test.

Not included:
- processed dataset ZIP archives;
- trained checkpoints;
- a complete multi-epoch training loop, checkpoint/resume manager, or final
  evaluation/inference command.

Do not report model performance until a complete training and evaluation run
has been carried out and its metrics have been saved.
