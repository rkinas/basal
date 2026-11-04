- **Audio is read up to the model's window, media work leaves the event loop** (`basal-serve --media on`,
  basal-omni): a clip was decoded whole before its length was checked, so a few hundred kB of FLAC silence could
  decode to gigabytes of float32; a clip is now read up to the window (30 s on Gemma 4; one frame more shows it is
  longer) and refused with the same 422 as before, and a clip of a model without a known window up to 67 million
  samples (`MAX_AUDIO_SAMPLES`, 256 MB of float32). Images and audio are decoded in a worker thread instead of the
  event loop, and the engine's media calls run in a thread of their own, so media requests waiting for the engine no
  longer take the threads text batches need. Answers are unchanged.
