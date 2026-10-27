# Results against basal-1.5

Measured once on the frozen release weights and published as measured, including where basal-2.0 is not better. Each
basal-2.0 model is compared with the basal-1.5 model of the same size. Full tables per model, with intervals and notes,
are on the model cards.

## Werdykt: hidden Polish–English decision benchmark

[Werdykt](https://basal.si5.pl/benchmark/) has 10 categories in Polish and English (5,000 decisions), one call per
item and one protocol for every model; its items are not published. Macro accuracy over the 20 cells:

| | basal-1.5-mini | **basal-2.0-mini** | basal-1.5 (4.5B) | **basal-2.0 (4.5B)** | basal-1.5-max | **basal-2.0-max** |
|---|---|---|---|---|---|---|
| macro [95% CI] | 0.607 [0.594, 0.619] | **0.618** [0.605, 0.631] | 0.721 [0.710, 0.733] | **0.729** [0.717, 0.739] | 0.773 [0.762, 0.783] | **0.784** [0.774, 0.795] |
| 2.0 − 1.5 | | +1.2 pt | | +0.7 pt | | +1.1 pt |
| PL / EN | 0.619 / 0.594 | 0.628 / 0.609 | 0.724 / 0.718 | 0.729 / 0.728 | 0.777 / 0.769 | 0.787 / 0.781 |
| calibration error (ECE, lower is better) | 0.086 | **0.035** | 0.068 | **0.060** | 0.074 | **0.044** |
| automated at 5% target error | 21.9% (4.33% err.) | 25.4% (5.12%) | 50.1% (5.04%) | 51.3% (5.08%) | 65.6% (4.99%) | 65.6% (4.51%) |
| automated at 1% target error | 11.9% (1.66% err.) | 15.2% (1.82%) | 22.5% (1.23%) | 11.9% (0.66%) | 42.5% (0.93%) | 35.3% (1.34%) |
| decisions/s at 16 clients, bf16, one H100 | 20.5 | **106.4** (×5.2) | 23.0 (SGLang) | **32.2** (×1.4) | 10.2 (SGLang) | **16.2** (×1.6) |
| latency at one client (p50) | 14 ms | 16 ms | 34 ms (SGLang) | 25 ms | 67 ms (SGLang) | 37 ms |

basal-1.5 rows are the public board's rows: basal-1.5 (4.5B) and basal-1.5-max served on SGLang (their fastest
configuration), basal-1.5-mini in basal-serve's default mode; basal-2.0 is served by basal-serve v2.0.0 in its default
mode. With both in basal-serve's default mode the 4.5B and max speed-ups are ×4.0 and ×3.5. Each release uses its own
calibration file; accuracy with `--deterministic`. *Automated at X% target error*: a confidence threshold chosen on one
half of the items for that error, applied to the other half (share automated, observed error in brackets). At the 1%
target basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.

**Where basal-2.0 stands.** On the public board hosted API models and larger open decision systems score above
basal-2.0; the [board](https://basal.si5.pl/benchmark/) lists every tested system with its accuracy, latency and cost.

## Accuracy by document length

Werdykt's 5,000 decisions by the length of the state (the document the question is about), in characters (about two
characters per token; 1,000 characters is about half a page). Plain accuracy per bucket, not the 20-cell macro; the
same items for every model:

| state length (characters) | items | 2.0-mini | 2.0 (4.5B) | omni-pico | omni-1.0-mini | omni-1.0-max | omni-1.1-max | omni-1.0-ultra |
|---|---|---|---|---|---|---|---|---|
| 0-1k | 709 | 49.6% | 61.6% | 54.3% | 66.4% | 72.2% | 73.3% | 77.9% |
| 1k-2k | 2,598 | 60.6% | 71.4% | 64.2% | 74.4% | 80.6% | 80.2% | 86.5% |
| 2k-4k | 796 | 73.4% | 85.3% | 76.8% | 91.1% | 94.0% | 93.2% | 97.6% |
| 4k-8k | 394 | 81.7% | 90.9% | 86.3% | 93.7% | 98.0% | 98.0% | 98.7% |
| 8k-16k | 128 | 64.1% | 72.7% | 68.8% | 79.7% | 85.2% | 84.4% | 91.4% |
| 16k+ | 375 | 47.5% | 59.2% | 54.9% | 65.1% | 70.7% | 70.1% | 83.5% |
| all | 5,000 | 61.8% | 72.9% | 66.0% | 76.9% | 82.3% | 82.1% | 87.9% |

The buckets hold different kinds of items (short states are mostly rules and abstention items), so a row is the
accuracy on that bucket's items, not the effect of length alone. basal-serve, bf16, one H100 (ultra: one H200);
basal-2.0 in `--mode fast`, basal-omni in `--mode eager`. Each model card adds 95% intervals, Werdykt's long-document
family (one task at 8k to 30k+ characters) and the held-out tests of its Results table by prompt length in tokens.

## Cascade: pico, mini, ultra

`basal-router` can ask basal-omni-pico first, basal-omni-1.0-mini where pico is unsure, and basal-omni-1.0-ultra where
mini is still unsure ([SERVING.md](SERVING.md), "Several levels"). From each model's answers on the held-out tests of
its card's Results table (the same items for all three), with one threshold T at both steps: pico answers when its top
probability is at least T, else mini does when its own is at least T, else ultra does. Accuracy of the answers given,
then the share of items each model answered (its accuracy on those in brackets):

| PL decisions | items | accuracy | pico answers | mini answers | ultra answers |
|---|---|---|---|---|---|
| pico alone | 7,081 | 87.6% | 100% | – | – |
| mini alone | 7,081 | 93.2% | – | 100% | – |
| ultra alone | 7,081 | 97.9% | – | – | 100% |
| cascade, T = 0.6 | 7,081 | 90.0% | 92.5% (90.4%) | 6.2% (83.6%) | 1.3% (90.2%) |
| cascade, T = 0.7 | 7,081 | 92.2% | 85.8% (92.5%) | 9.7% (89.8%) | 4.5% (90.9%) |
| cascade, T = 0.8 | 7,081 | 94.0% | 77.9% (94.5%) | 12.5% (93.3%) | 9.6% (91.2%) |
| cascade, T = 0.9 | 7,081 | 96.1% | 64.6% (97.0%) | 16.7% (95.5%) | 18.8% (93.5%) |

| EN corpus test | items | accuracy | pico answers | mini answers | ultra answers |
|---|---|---|---|---|---|
| pico alone | 1,479 | 60.7% | 100% | – | – |
| mini alone | 1,479 | 67.5% | – | 100% | – |
| ultra alone | 1,479 | 75.1% | – | – | 100% |
| cascade, T = 0.6 | 1,479 | 67.3% | 67.8% (73.4%) | 12.1% (62.6%) | 20.1% (49.8%) |
| cascade, T = 0.7 | 1,479 | 68.8% | 59.2% (77.0%) | 12.8% (64.2%) | 28.0% (53.4%) |
| cascade, T = 0.8 | 1,479 | 70.9% | 49.0% (81.2%) | 13.0% (74.5%) | 38.0% (56.4%) |
| cascade, T = 0.9 | 1,479 | 73.1% | 37.3% (86.1%) | 13.0% (79.8%) | 49.6% (61.6%) |

| PL-Real | items | accuracy | pico answers | mini answers | ultra answers |
|---|---|---|---|---|---|
| pico alone | 300 | 94.0% | 100% | – | – |
| mini alone | 300 | 95.0% | – | 100% | – |
| ultra alone | 300 | 97.3% | – | – | 100% |
| cascade, T = 0.6 | 300 | 95.0% | 98.0% (95.2%) | 1.7% (5 items: 80%) | 0.3% (1 item: 100%) |
| cascade, T = 0.7 | 300 | 94.7% | 97.3% (95.5%) | 1.7% (5 items: 60%) | 1.0% (3 items: 67%) |
| cascade, T = 0.8 | 300 | 95.3% | 94.0% (96.8%) | 4.7% (14 items: 71%) | 1.3% (4 items: 75%) |
| cascade, T = 0.9 | 300 | 97.0% | 88.0% (98.9%) | 7.3% (22 items: 86%) | 4.7% (14 items: 79%) |

Computed from the answers each model gave alone (the readouts behind its card: basal-serve, bf16, both option orders
averaged, as the Results tables), not from a run through the router; the router uses the top probability of the answer
as the replica returns it, so with one option order the shares move somewhat. The items ultra answers are the ones the
smaller models were unsure of, so its accuracy on them is below its accuracy alone. No speed or cost is claimed here:
measure the share each level answers on your own traffic.

## Other benchmarks

Both models served by engine v2.0.0 on the same items; differences in points, paired, with 95% bootstrap intervals on
the model cards (* = the interval excludes 0).

| benchmark | mini: 1.5 → 2.0 | 4.5B: 1.5 → 2.0 | max: 1.5 → 2.0 |
|---|---|---|---|
| Decision Index (JDI 0.2.1) | 16.41 → 26.69 | 27.65 → 39.86 | – → 44.07 (no 1.5 index) |
| JDI requests left unanswered | 17.2% → 0.02% | 17.2% → 0.00% | – → 0.00% |
| LEK, Polish medical licensing exam, PL / EN (% correct) | 47.7 / 48.4 → 47.7 / 49.2 | 60.5 / 59.7 → 62.0 / 63.0 | 70.0 / 67.6 → 75.6 / 74.1 |
| PL-Holdout (4,000 items) | −0.8 | +0.6 | −0.6 |
| Polish decisions (7,081 items) | −1.4* | +0.1 | −0.2 |
| PL-Real-300, doc-clean subset (158 items) | 0.0 | 0.0 | +1.3 |
| PL-Real-300, all 300 items | −2.0 | −2.0 | 0.0 |
| English panel (10 public tasks, mean accuracy) | −2.1* | −3.6* | −1.8* |
| Polish general (3,079 items; development) | +2.2* | +1.6* | +1.4* |
| English corpus test (1,479 items; development) | +1.1 | +0.1 | +1.2 |

basal-1.5 answers questions with at most 10 options, so it leaves 17.2% of JDI requests unanswered. Polish general and
the English corpus test were used to choose between training runs, so their numbers are development numbers; the other
benchmarks were not used to select basal-2.0. LEK questions: NASK-PIB Reassess, 2008–2026 sessions (CC BY-NC-SA 4.0),
shown for the basal models only. The PL-Real primary figure is the subset whose source documents no basal training row
came from.

## Speed

One decision = one question, one option order; bf16 unless noted.

| | mini | 4.5B | max |
|---|---|---|---|
| bf16, one H100, Werdykt speed pass: decisions/s at 16 clients; p50 at one client | 106.4; 16 ms | 32.2; 25 ms | 16.2; 37 ms |
| `-FP8`, one H100, port check: decisions/s at 16 clients; p50 at one client | 103.5; 14 ms | 38.6; 29 ms | 19.9; 50 ms |
| `-MLX-8bit`, Apple M4 Pro 24 GB: one question / five questions over one state (p50) | 93 ms / 223 ms | 278 ms / 698 ms | 622 ms / 1,506 ms |

The bf16 and FP8 rows come from different workloads (the Werdykt items and the port check's development items), so
compare within a row, not across them.

Engine v2.0.0 items, each left out in turn (one H100, bf16): compact batching +19% decisions/s at 16 clients, +17% at
32, +8% at one client; the prompt-template cache +20% at 8 clients (mini and max; off for the 4.5B model, see the quick
start). In a mixed load at 32 clients the scheduler answers short requests in 0.81 s instead of 1.38 s (p50) while long
requests take about 7% longer (1.96 s instead of 1.83 s); `--scheduler off` serves in arrival order.

basal-1.5 also shipped FP8 checkpoints served by vLLM, which on long, repeated documents under mixed traffic were much
faster than basal-1.5 in bf16 (model cards, "basal-1.5's other configuration"). basal-2.0 was not measured on that
workload: for such traffic, measure both on your own load before you switch.
