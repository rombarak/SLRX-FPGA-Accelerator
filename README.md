# SLRX FPGA Accelerator

> **2nd place — HackaNuvoTon 2026 hackathon** · Faculty of Engineering,
> Bar-Ilan University · 
> Awarded for developing and implementing an integrated digital hardware accelerator on an FPGA SoC for
> an artificial-intelligence sign-language-recognition system.

A hardware-accelerated convolutional neural network for **sign-language character recognition**,
built for the Kuntz-5 SoC + xfarm accelerator platform and synthesized for an **Intel MAX10 (10M50,
DE10-Lite)** FPGA. The neural-network layers run as custom **SystemVerilog** accelerators (`hw/`),
driven by **C** firmware (`sw/`) that configures them over host registers and shared memory.

> **Result:** the per-character workload was taken from a **780,855-cycle software baseline** down to
> **897 hardware cycles @ 75.9 MHz (ITT ≈ 11.82 µs)** — a **~870× reduction in cycles** — at
> **100 % classification accuracy (20 / 20 test images)**, verified on the physical FPGA.

---

## Table of contents
1. [The task and the network](#1-the-task-and-the-network)
2. [How the accelerator works](#2-how-the-accelerator-works)
3. [How it was built and optimized](#3-how-it-was-built-and-optimized-the-process)
4. [Final results](#4-final-results)
5. [Repository layout](#5-repository-layout)
6. [Scope](#6-scope)

---

## 1. The task and the network

The system classifies a **32×32 grayscale** hand image into one of **27 sign-language characters**.
It is a small CNN, run end-to-end on the FPGA:

```
 32×32  ──Conv0 5×5 (no pad)──▶ 28×28 ──MaxPool 2×2──▶ 14×14      (ConvPool0)
 14×14  ──Conv1 5×5 (no pad)──▶ 10×10 ──MaxPool 2×2──▶  5×5       (ConvPool1)
  5×5 = 25 ──flatten──▶ FC0 (25→32) ──▶ FC1 (32→27) ──▶ arg-max ──▶ label
```

**Data types & arithmetic (must match a C reference bit-for-bit):**
- Feature-maps: `uint8` [0, 255]; convolution/linear weights: `int8`; conv bias: scalar `int32`;
  linear bias: `int32` vector.
- Every layer is followed by `relu_and_descale(x)` = **ReLU**, then **`x / 256`** (descale), then
  **saturate to 255**.
- A convolution output = `bias + Σ (uint8 input × int8 weight)` over the 5×5 window, then
  `relu_and_descale`.
- 2×2 max-pool = `max` over the 2×2 block of conv outputs.

---

## 2. How the accelerator works

### 2.1 CPU ↔ accelerator interface
The CPU and the accelerators communicate through **host registers** and a **shared 256-bit-wide
memory (XMEM)**:
1. The CPU writes the operand addresses, dimensions, and a **command** into host registers.
2. It writes the command id to `XLR_START_RI`; each accelerator decodes it and **only the addressed
   accelerator responds**.
3. The accelerator raises `xlr_done` and holds it until the CPU reads it; the CPU polls
   `while (!HOST_REG(XLR_DONE_RI)) {}`.

The accelerators are **hardware-looped**: a *single* command computes an *entire* layer (the loop over
all output positions lives in hardware), so the CPU issues one command and polls once per layer.

### 2.2 Fused Conv + MaxPool — `hw/conv/conv.sv`
Instead of computing a full convolution feature-map, storing it, and pooling it in a second pass, one
`CONV_WINDOW` command computes the **pooled** output directly:
- Each **pooled pixel** is the 2×2 max of **four** 5×5 convolution windows (each a 25-tap
  multiply-accumulate). The four windows span **6 input rows**, held in an on-chip **line buffer**.
- **Four parallel MAC windows** compute the four convolutions; a **4-stage pipeline**
  (*select → multiply → combine + activate → max-pool*) keeps every combinational path short so the
  multipliers pack into MAX10 DSP blocks — the deeper pipeline actually **raises Fmax**.
- A **rolling line buffer** reuses 4 of the 6 input rows between output rows and fetches only the 2 new
  ones; a **prefetch** engine reads those 2 rows *during* the current row's compute, and a
  **write-overlap** engine flushes the finished row *during* the next row's compute — so almost all
  memory traffic is hidden behind computation.

### 2.3 Pipelined Linear + fused Select — `hw/linear/linear.sv`
- The whole bias vector is **bulk-read once**; each output element **streams one weight row** and runs a
  **32-tap MAC**; the full output vector is **block-written** in one transaction.
- The 32-MAC is a **4-stage pipeline** with a **balanced adder tree**, which lifts the linear
  accelerator to ~80 MHz.
- The final **arg-max ("Select") is computed on-chip**, one stage behind the MAC, so the winning label
  is produced with no separate software pass.
- Weight-row reads use **back-to-back latency-1** transfers (a new address every cycle), matched to the
  memory port's one-cycle read latency.

---

## 3. How it was built and optimized (the process)

### 3.1 Methodology
- **Golden model first.** Every operation was written first as a C reference (`*_nox` functions) and
  used as a **bit-exact oracle**. Hardware was only accepted once it matched the reference in
  simulation (per-pixel self-checks) and the detected string was unchanged.
- **Incremental, reversible steps.** Each optimization was a small, self-contained increment behind a
  compile-time toggle, with a known-good backup kept at every step, so any change was one move away from
  a working fallback.
- **Cloud is ground truth.** Synthesis, fitting, static-timing analysis (STA), and the cycle
  measurement run on a remote toolchain; every claim of "faster / fits / meets timing" was measured
  there, never assumed. Cycle counts come from a single-iteration run; Fmax from `qsyn ... -sta`.

### 3.2 Phase 1 — from software to basic hardware acceleration (10.8×)
The project started as a **purely software** inference at **780,855 cycles per character** — the first
convolution alone (Conv0) was **~636,598 cycles (81.5 %)**. The first milestone moved every layer into
custom accelerators and handed the loop control to hardware.

**Hardware-looped, autonomous accelerators.** Instead of the CPU driving each output element and
handshaking with the accelerator every step, the CPU issues a *single* command (e.g. `LIN_CALC`) for the
whole matrix. On the start pulse the accelerator becomes fully autonomous: an internal state-machine
runs a self-advancing counter, computes the next memory addresses, fetches the weight column and bias,
performs the multiply-accumulate, writes the result, and immediately advances to the next element —
with **zero dead cycles of CPU communication**.

**Two changes made the conv accelerator robust and synthesizable:**
- **Dynamic read pointer (universal dimensions).** Early versions hard-coded a 32-byte row step
  (assuming a 32×32 input); at Conv1 (14×14) this read out-of-bounds, never received `mem_valid`, and
  deadlocked the FSM. A pointer that increments by the *current* layer's dimension made the module
  parameter-agnostic across every convolutional layer.
- **Barrel-shifter datapath (routing congestion).** Indexing a 2-D line buffer by a variable column
  synthesized thousands of large multiplexers → routing congestion and Fitter crashes. Restructuring the
  buffers into **256-bit packed rows** and **shifting the whole row right by `col × 8` bits** maps onto
  the FPGA's efficient barrel shifters, eliminating the congestion so the design passes synthesis and fit.

**Phase-1 result — software vs. basic acceleration, per layer (cycles):**

| Layer | SW only | HW-accelerated | Speed-up |
|---|---:|---:|---:|
| Conv0 | 636,598 | 57,483 | 11.1× |
| Pool0 | 10,358 | 1,171 | 8.8× |
| Conv1 | 64,878 | 7,747 | 8.4× |
| Pool1 | 1,076 | 595 | 1.8× |
| Linear0 | 32,440 | 2,363 | 13.7× |
| Linear1 | 34,798 | 2,043 | 17.0× |
| Select | 707 | 1,211 | 0.6× |
| **Total** | **780,855** | **72,613** | **10.8×** |

### 3.3 Phase 2 — architectural optimization (72,613 → 897 cycles)
Basic acceleration still computed a full-resolution convolution map and touched memory on almost every
operation. Phase 2 fused and pipelined the datapath to eliminate both: **fused Conv + MaxPool** (compute
the pooled map directly), an on-chip **line buffer** (each input row is read once and reused),
a **4-stage MAC pipeline**, **rolling buffer + prefetch**, **back-to-back reads**, **write/compute
overlap**, a **pipelined 32-MAC linear** with a **balanced adder tree**, and a **fused on-chip Select**.
This cut the workload a further **~81×**, to the final **897 cycles @ 75.9 MHz (ITT ≈ 11.82 µs)**.

ITT across the fused-optimization sub-steps:

| Milestone | ITT |
|---|---|
| First fused hardware version | ≈ 71 µs |
| 2 MACs + streaming + rolling buffer | ≈ 42 µs |
| 4-MAC pipeline + fused Select | ≈ 23.6 µs |
| + prefetch + back-to-back reads | ≈ 17.9 µs |
| + 32-MAC pipeline + balanced adder tree | ≈ 12.1 µs |
| + write/compute overlap | **≈ 11.82 µs** |

**Engineering insights that drove the numbers**
- *Parallelism cuts cycles, but only with pipelining.* A flat 4-window multiplier in one cycle failed
  timing; splitting it into registered stages both met timing **and** raised Fmax by ~30 %.
- *The DSP budget is the ceiling.* MAX10 has 144 multiplier blocks; 4 conv windows (~100 multipliers)
  is the sweet spot — 8 windows overflowed the DSPs and lost Fmax.
- *Hide memory behind compute.* The compute pipeline never touches the memory port while it runs, so the
  port is used in parallel to prefetch the next rows and flush the previous result.
- *The shared memory port is a latency-1 pipeline.* Data returns one cycle after the address is
  presented; binding each transfer to a lagging index enables single-cycle back-to-back reads.
- *A ripple adder chain is a sequential bottleneck.* Rewriting the MAC's summation as a balanced tree
  (depth 7 → 3) was bit-exact and gave the single largest Fmax jump.

---

## 4. Final results

| Metric | Value |
|---|---|
| Software baseline (received) | 780,855 cycles/char |
| Basic hardware acceleration | 72,613 cycles/char (**10.8×** vs software) |
| **Final (fused + pipelined)** | **897 cycles/char** (**~870×** vs software) |
| System clock (Fmax) | **75.9 MHz** (clock target 50 MHz — beaten by +52 %) |
| **ITT = cycles / Fmax** | **≈ 11.82 µs** |
| Classification accuracy | **100 % (20 / 20)** |

**Final per-layer cycle breakdown (fused build)**

| Stage | Cycles |
|---|---:|
| ConvPool0 (fused Conv+MaxPool, 32×32 → 14×14) | 379 |
| ConvPool1 (fused Conv+MaxPool, 14×14 → 5×5) | 143 |
| Linear0 (25 → 32) | 164 |
| Linear1 (32 → 27) | 188 |
| Select (on-chip arg-max) | 23 |
| **Total** | **897** |

**Timing & resources** — conv accelerator 73.4 MHz, linear accelerator 80.5 MHz, system 75.9 MHz;
≈ 13 K of ~50 K logic elements, DSP within the device's 144; Intel MAX10 10M50 (DE10-Lite).
Verified in RTL simulation and on the physical FPGA against a C golden reference model.

---

## 5. Repository layout
```
hw/conv/conv.sv        fused Conv + MaxPool accelerator
hw/linear/linear.sv    pipelined Linear + fused Select accelerator
hw/top/slrx_enums.svh  shared CPU ↔ accelerator register / command interface
sw/                    C drivers (conv, linear, pool, top-level inference + arg-max)
bitstream/             k5_xbox_slrx897.sof — the verified FPGA bitstream for the results above
```

## 6. Scope
This repository contains the **accelerator RTL, C drivers, documentation, and the verified bitstream**.
Course/platform-provided files (the platform top-level, the reference pooling module, the Quartus /
simulation project, and course materials) are intentionally **not** included, so the design is not
meant to build standalone without that platform.
