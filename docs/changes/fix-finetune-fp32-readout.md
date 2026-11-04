- **basal-finetune's training steps read the codes in fp32** (`basal-finetune`, bf16 / fp16 training): under the
  training steps' autocast the code readout (the output head's input times the code rows) was a bf16 product, so the
  loss saw logits rounded to bf16 steps (0.125 at 16-32); it is fp32 now, as `basal-serve` reads them. The accuracy
  before and after training was already read in fp32.
