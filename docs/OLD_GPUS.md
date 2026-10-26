# Old and non-NVIDIA GPUs: GGUF with llama.cpp's Vulkan backend

basal's own engine needs a recent NVIDIA GPU (or an Apple chip). A GTX 10xx, an RX 480 / 580 or an Intel Iris Xe
can still serve the models as **GGUF files through llama.cpp's Vulkan backend**: Vulkan runs on almost every GPU
with a driver from the last ten years, whatever its maker. This page shows the setup and a self-check that tells you
whether your card computes the same decisions as a reference.

**Status (2.5): tested on one Vulkan device only, an Apple M4 Pro through MoltenVK.** Nobody has run this path on a
GTX 10xx, a Radeon RX or an Iris Xe yet; [What was tested](#what-was-tested) says exactly what was and what was not.
Treat the numbers below as the expected result, and report what you get ([Report back](#report-back)).

| card | why not the basal engine | Vulkan driver |
|---|---|---|
| GTX 10xx (Pascal) and older NVIDIA | no tensor cores, no bf16, and recent PyTorch CUDA builds no longer ship kernels for it (check: `python -c "import torch; print(torch.cuda.get_arch_list())"`) | the NVIDIA driver |
| RX 480 / 580 (Polaris) and older AMD | not supported by ROCm | Mesa RADV (Linux), AMD Adrenalin (Windows) |
| Intel Iris Xe and other integrated GPUs | no CUDA, shares system memory | Mesa ANV (Linux), Intel driver (Windows) |

## 1. llama.cpp with Vulkan

Take a prebuilt Vulkan build from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (the
`...-bin-win-vulkan-x64.zip` for Windows or `...-bin-ubuntu-vulkan-x64.zip` for Linux), or build it:

```bash
# Linux: the Vulkan loader, headers and the shader compiler (Ubuntu: libvulkan-dev glslc), and a driver (mesa-vulkan-drivers)
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server
./build/bin/llama-server --list-devices      # must list "Vulkan0: <your card>"
```

Windows: [llama.cpp's build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan) (Vulkan SDK
and w64devkit or MSVC). If `--list-devices` shows no Vulkan device, the driver is the problem, not llama.cpp
(`vulkaninfo --summary` shows what the system sees). Use a recent driver: the shaders use features that very old
drivers lack or implement wrongly, which is what the self-check below catches.

## 2. Which file

Files are in the `-GGUF` repositories. `Q8_0` is the one to start with; `Q5_K_M` saves a third of the memory at a
small cost in agreement ([QUANTIZATION.md](QUANTIZATION.md)). The model needs its file in memory plus the context
(about 0.2 GB for 4,096 tokens on the small models) and the compute buffers (0.1 to 0.5 GB):

| model | file | size | fits in | 
|---|---|---|---|
| basal-omni-pico (0.8B) | `basal-omni-pico-Q8_0.gguf` | 0.8 GB | 2 GB cards, Iris Xe with 4 GB shared |
| basal-2.0-mini (1.5B) | `basal-2.0-mini-Q5_K_M.gguf` / `-Q8_0.gguf` | 1.2 / 1.7 GB | 2 GB / 4 GB |
| basal-2.0 (4.5B) | `basal-2.0-4.5B-Q5_K_M.gguf` / `-Q8_0.gguf` | 3.4 / 5.1 GB | 4 GB / 6 GB (GTX 1060 6 GB, RX 580 8 GB) |
| basal-2.0-max (11B) | `-Q5_K_M.gguf` / `-Q8_0.gguf` | 8.0 / 11.9 GB | 12 GB cards; less with `-ngl` (below) |

Sizes are from the repositories; the "fits in" column is our estimate from them, not a measurement on those cards.
With less memory than the table asks for, offload part of the layers (`-ngl 20`): the rest runs on the CPU, slower.

## 3. Serve

```bash
hf download Remek/basal-omni-pico-GGUF --local-dir basal-omni-pico-GGUF
llama-server -m basal-omni-pico-GGUF/basal-omni-pico-Q8_0.gguf -c 8192 -np 1 -ngl 99 --port 8080
basal-serve --model ./basal-omni-pico-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

`basal-serve --mode llamacpp` sends the engine's token ids to `llama-server` ([INSTALL.md](INSTALL.md#ollama-and-llamacpp-gguf));
`-c` is the longest prompt it reads (longer states are cut in the middle: `usage.truncated`). `-ngl 99` puts every layer
on the GPU. The basal side needs no GPU: `pip install basal` on any machine, CPU only.

If `llama-server` prints wrong answers or `nan`, try, one at a time, the environment variables of the Vulkan backend:
`GGML_VK_DISABLE_F16=1` (drivers with broken half-precision shaders), `GGML_VK_DISABLE_COOPMAT=1`,
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`; `GGML_VK_VISIBLE_DEVICES=0` picks the GPU when the machine has several (an
integrated and a discrete one). They are llama.cpp's, not basal's; tell us which one fixed it.

## 4. Self-check

`basal-selfcheck` sends 20 fixed requests (60 questions: short, medium and long states, Polish and English, choice /
yes-no / score) to the server and compares the probabilities with a **reference file shipped with basal**. It prints
the top-1 agreement and the largest change of any option's probability (`max |dp|`). Start `basal-serve` with
`--no-calibration`, as the references were made:

```bash
basal-serve --model ./basal-omni-pico-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --no-calibration --port 8000
basal-selfcheck --url http://127.0.0.1:8000 --reference basal-omni-pico-Q8_0 --note "GTX 1060 6GB, Vulkan, driver 5xx"
```

Two references per model. Use the first for the GPU, the second to see what the quantisation costs:

| `--reference` | made with | answers |
|---|---|---|
| `basal-omni-pico-Q8_0` | the same GGUF file on llama.cpp's **CPU** backend | does my GPU compute what the CPU computes from this file? |
| `basal-omni-pico` | basal's PyTorch engine, fp32 | how far is the quantised file from the original model? |

(The shipped references are for basal-omni-pico; `basal-selfcheck --make-reference FILE` makes one for another model or
file from a server you trust.)

**How to read it.**

- *top-1 outside near ties* is the share of questions where your server's most probable option is the reference's. A
  flip between two options the reference had within 0.02 of each other counts as agreement. The check passes at
  0.98 or more (`--gate`), and exits 0 or 1.
- *max |dp|* against the CPU reference of the same file: the Apple M4 Pro through Vulkan gave 0.04 (mean 0.003).
  Values of that size are rounding differences between GPU and CPU. A GPU that is wrong shows `max |dp|` of 0.1 and
  more, top-1 below 0.9 or non-finite answers (an error on the request, which the check counts as failed).
- *max |dp|* against the fp32 reference is larger and normal for a quantised file: 0.35 for this file's Q8_0
  (mean 0.06), with the top option unchanged on all 60 questions. The model card of the GGUF repository gives the
  agreement on a large set.
- Requests the server had to cut (`usage.truncated`, the context was shorter than the prompt) are left out and counted;
  raise `-c`.
- The seconds per request are for your orientation only.

## What was tested

- **Tested:** llama.cpp (`9125a2e`, 26 Oct 2026) built with `-DGGML_VULKAN=ON` on macOS, Vulkan device
  `Apple M4 Pro` through MoltenVK (the Vulkan-on-Metal layer), `basal-omni-pico-Q8_0.gguf`, all 25 layers offloaded
  (`-ngl 99`), through `basal-serve --mode llamacpp`:

  | compared with | readouts | top-1 outside near ties | mean / max \|dp\| |
  |---|---|---|---|
  | the same file on llama.cpp's CPU backend | 60 | 1.000 (top-1 1.000) | 0.0030 / 0.0387 |
  | the same file on llama.cpp's Metal backend (brew build 11146) | 60 | 1.000 | 0.0614 vs fp32 for both: Metal and Vulkan differ from fp32 alike |
  | basal's fp32 engine | 60 | 1.000 | 0.0622 / 0.3547 |

- **Not tested:** any native Vulkan driver (NVIDIA, Mesa RADV or ANV, AMD or Intel Windows drivers), a GTX 10xx,
  an RX 480 / 580, an Iris Xe, Windows, or a Linux build of the Vulkan backend. MoltenVK is a translation layer to
  Metal, so a pass there shows that llama.cpp's Vulkan shaders and basal's llama.cpp mode work together on this model,
  not that your driver does. The models other than basal-omni-pico were not run through Vulkan.
- The basal-omni-pico GGUF is a hybrid model (Gated DeltaNet and attention layers). `basal-selfcheck` on the Metal backend of
  llama.cpp passes as well.

## Report back

A run takes a few minutes. For each card send us (the author collects them):

1. the output of `llama-server --list-devices` and the first lines of its log (`ggml_vulkan: ...` names the device,
   driver and which of F16 / coopmat / integer dot it uses);
2. the full output of `basal-selfcheck` for `basal-omni-pico-Q8_0` and, if the card has the memory, `basal-2.0-mini`
   (both references; a reference for a model that is not shipped is not needed);
3. the operating system, the driver version, and whether any `GGML_VK_*` variable was needed;
4. `PASS` or `FAIL`, and for `FAIL` the lines that show it.
