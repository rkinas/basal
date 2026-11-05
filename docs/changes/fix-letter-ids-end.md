- **A long first request no longer stalls basal-serve** (`basal-serve`, prompt format 2 most): the token ids of the
  option codes, computed once per server on its first prompt, were read by tokenizing that whole prompt once per code
  (255 codes with prompt format 2, 10 with format 1), so a first request with a long state held the event loop: 0.6 s
  with a 16 KB state, 3-6 s with 64 KB, 1.5-2.5 minutes with 1 MB on basal-2.0 and basal-omni tokenizers, with every
  other client and `/health` waiting. The codes are now read at the prompt's end, where its tokens are those of the
  whole prompt (the same ids on every basal tokenizer). Answers are unchanged.
