# Reports from other machines

What testers measured with `janas-bench` on their own machines, as they reported it in the issues of this repository. Every figure is theirs, from the version of Janas they ran at the time (the date says which): the engine has changed since, so rows of different dates are not a comparison between machines. Prompt and reply are tokens per second; "with MTP" is a chat reply with the drafts of the model's prediction block, greedy / sampled where the report gave both. Rows up to 26 September give the best configuration of each column; from 27 September, the configuration the engine itself chose, as a chat runs.

How to add your machine: run `janas-bench <model> --share`. When the measure ends it prints the address of a new issue with the report filled in, and opens it in your browser if you say yes. Nothing is sent until you send the issue yourself. More on running it well: [Helping it run on more hardware](hardware.md#helping-it-run-on-more-hardware).

[Every report, by model](#every-report-by-model) ranks every machine for each model, from the reports pasted in the issues. The table below it was checked by hand, row by row, up to 17 October 2026.

| Issue | Date | CPU | System | GPU | RAM | Model | Prompt | Reply | With MTP | Answer |
|---|---|---|---|---|---|---|---|---|---|---|
| [#1](https://github.com/prabanta-dev/janas/issues/1) | 20 Sep | Intel Core i7-1365U | Linux | not used | 30.7 GiB | Qwen3.6-35B-A3B | 23.1 | 10.8 | - | - |
| [#2](https://github.com/prabanta-dev/janas/issues/2) | 22 Sep | AMD Ryzen AI 9 HX 370 | Linux | not used | 23.1 GiB | Qwen3.6-35B-A3B | 44.8 | 16.0 | 24.6 | - |
| [#8](https://github.com/prabanta-dev/janas/issues/8) | 25 Sep | Intel Core Ultra 9 185H | Linux | Arc (integrated) | 30.8 GiB | Qwen3-4B | 89.7 | 26.1 | - | ok |
| [#8](https://github.com/prabanta-dev/janas/issues/8#issuecomment-5844758197) | 26 Sep | Intel Core i7-1365U | Linux | none | 30.7 GiB | Qwen3-4B | 40.3 | 10.8 | - | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3-4B | 341.5 | 21.1 | - | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3.6-35B-A3B | 317.4 | 22.8 | 49.5 | ok |
| [#10](https://github.com/prabanta-dev/janas/issues/10) | 26 Sep | AMD Ryzen 9 9950X3D | Linux | not used | 60.2 GiB | Qwen3-Next-80B-A3B | 143.4 | 16.7 | 20.7 | - |
| [#12](https://github.com/prabanta-dev/janas/issues/12) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3-4B | 366.4 | 21.5 | - | ok |
| [#13](https://github.com/prabanta-dev/janas/issues/13) | 27 Sep | Intel Core i9-14900HX | Linux | RTX 4070 Laptop | 62.5 GiB | Qwen3-Next-80B-A3B | 122.0 | 25.5 | 33.2 | ok |
| [#13](https://github.com/prabanta-dev/janas/issues/13) | 27 Sep | Intel Core i9-14900HX | Linux | not used | 62.5 GiB | Qwen3-235B-A22B (142 GB) | 8.2 | 3.0 | - | - |
| [#14](https://github.com/prabanta-dev/janas/issues/14) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3.6-35B-A3B | 338.9 | 22.1 | 43.3 / 44.0 | ok |
| [#15](https://github.com/prabanta-dev/janas/issues/15) | 27 Sep | AMD Ryzen 9 9950X3D | Linux | RTX 5080 | 60.2 GiB | Qwen3.6-35B-A3B | 374.6 | 23.0 | 45.5 / 45.5 | ok |
| [#16](https://github.com/prabanta-dev/janas/issues/16) | 13 Oct | AMD Ryzen 7 255 | Linux | Radeon 780M (integrated) | 31.2 GiB | Qwen3-4B | 477.4 | 20.5 | - | - |
| [#16](https://github.com/prabanta-dev/janas/issues/16) | 13 Oct | AMD Ryzen 7 255 | Linux | Radeon 780M (integrated) | 31.2 GiB | Qwen3-Next-80B-A3B | 61.6 | 16.2 | 23.7 / 25.8 | - |
| [#23](https://github.com/prabanta-dev/janas/issues/23) | 13 Oct | AMD Ryzen 9 5900HX | Linux | not built | 30.8 GiB | Qwen3-Next-80B-A3B | 28.2 | 9.5 | 16.6 / 20.7 | see note |
| [#37](https://github.com/prabanta-dev/janas/issues/37) | 17 Oct | Intel Core i5-1135G7 | Linux | Iris Xe (integrated) | 15.4 GiB | Qwen3-4B | 180.9 | 16.0 | - | ok |
| [#39](https://github.com/prabanta-dev/janas/issues/39) | 17 Oct | Intel Core i5-1135G7 | Linux | Iris Xe (integrated) | 15.4 GiB | Qwen3-Next-80B-A3B | 15.7 | 7.3 | 11.0 / 11.2 | ok |
| [#40](https://github.com/prabanta-dev/janas/issues/40) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3-4B | 30.9 | 25.0 | - | - |
| [#41](https://github.com/prabanta-dev/janas/issues/41) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3.5-9B | 23.7 | 15.2 | 21.0 / 21.2 | - |
| [#42](https://github.com/prabanta-dev/janas/issues/42) | 17 Oct | AMD Ryzen 7 7840HS | Windows 11 | RTX 4070 Laptop | 31.3 GiB | Qwen3-Next-80B-A3B | 26.3 | 13.7 | 23.5 / 27.2 | - |

## Notes

- **Answer**: "ok" where `janas-try` found the model's answer at temperature 0 to be the very text every other machine gets, "-" where the report did not check it.
- **#13**: Qwen3-235B-A22B, 142 GB, on 62.5 GiB of RAM, with an expert cache of 38.7 GiB: 86.9% of the experts found in memory, the rest read from the NVMe disk.
- **#15**: with `JANAS_GPU_MIN_N=1` the RTX 5080 also took a reply's single tokens and Qwen3-4B replied at 25.9 tok/s instead of 22.4; since then the engine tries this on any discrete GPU and keeps it where it measures faster.
- **#16**: the integrated GPU read the prompt 5.2 times faster than the CPU alone (477 against 92 tok/s with `JANAS_GPU_FAST_PROMPT=0`).
- **#23**: the answer differed on every machine, not only on this one: since 1 October `janas-chat` wrote the catalog of its services into the system message, and `janas-try` expected the answer from before. Fixed on 14 October.
- **#39**: a 48 GB model on 15.4 GiB of RAM, an expert cache of 7.0 GiB.
- **#40-#42**: the prompt reads far slower than on the other machines; being looked into with the tester.

## Every report, by model

Every `janas-bench` report pasted into an issue or a comment of this repository, the project's own left out, for each model the fastest reply first. The figures are the configuration the engine chose on that machine (the report's `result:` line): reply and prompt in tokens a second, and with MTP the greedy and the sampled reply where the model has a prediction block. Machines differ in more than the CPU and the GPU named here: the disk, the memory free at the time, the power, and the version of Janas, which the "Janas" column gives. Made by `tools/reports_table` from the issues; a report attached as a file rather than pasted is not read.

<!-- reports:begin -->

### Gemma-4-12B-it

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 8.0 | 41.5 | 14.0 / 14.7 | Intel Core Ultra 5 125U | Intel Graphics | 30.7 GiB | Linux | 6b8ba9c | [#72](https://github.com/prabanta-dev/janas/issues/72), 13 Nov 2026 |

### Qwen3-4B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 93.7 | 834.2 | - | AMD Ryzen 5 8500G | NVIDIA GeForce RTX 3060 | 30.5 GiB | Linux | v2026.11.03 | [#54](https://github.com/prabanta-dev/janas/issues/54), 3 Nov 2026 |
| 29.9 | 220.8 | - | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.18-dirty | [#40](https://github.com/prabanta-dev/janas/issues/40#issuecomment-6249248529), 18 Oct 2026 |
| 28.2 | 45.2 | - | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.18-dirty | [#40](https://github.com/prabanta-dev/janas/issues/40#issuecomment-6249248529), 18 Oct 2026 |
| 26.6 | 281.3 | - | AMD Ryzen 7 5700 | Radeon RX 5500 XT | 15.8 GiB | Windows | v2026.10.14 | [#36](https://github.com/prabanta-dev/janas/issues/36), 16 Oct 2026 |
| 25.2 | 45.2 | - | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.18-dirty | [#40](https://github.com/prabanta-dev/janas/issues/40#issuecomment-6249248529), 18 Oct 2026 |
| 25.0 | 30.9 | - | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.14 | [#40](https://github.com/prabanta-dev/janas/issues/40), 17 Oct 2026 |
| 22.4 | 261.4 | - | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 2f34204 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-5859891396), 27 Sep 2026 |
| 22.4 | 260.2 | - | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 2f34204 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-5859891396), 27 Sep 2026 |
| 22.3 | 263.6 | - | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 2f34204 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-5859792985), 27 Sep 2026 |
| 22.3 | 263.8 | - | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 2f34204 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-5859792985), 27 Sep 2026 |
| 21.8 | 738.0 | - | AMD Ryzen 9 9950X3D | none | 60.2 GiB | Linux | v2026.11.04 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-6642807855), 4 Nov 2026 |
| 21.8 | 488.5 | - | AMD Ryzen 9 9950X3D | none | 60.2 GiB | Linux | v2026.11.04 | [#15](https://github.com/prabanta-dev/janas/issues/15#issuecomment-6642807855), 4 Nov 2026 |
| 21.6 | 151.8 | - | AMD Ryzen 5 8500G | none | 30.5 GiB | Linux | v2026.11.03 | [#53](https://github.com/prabanta-dev/janas/issues/53), 3 Nov 2026 |
| 21.5 | 366.4 | - | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 61de67d | [#12](https://github.com/prabanta-dev/janas/issues/12), 27 Sep 2026 |
| 21.3 | 291.4 | - | AMD Ryzen 5 7640HS | AMD Radeon 760M Graphics | 22.8 GiB | Linux | v2026.10.30.1-dirty | [#51](https://github.com/prabanta-dev/janas/issues/51), 1 Nov 2026 |
| 21.1 | 341.5 | - | AMD Ryzen 9 9950X3D | none | 60.2 GiB | Linux | 791b07d | [#11](https://github.com/prabanta-dev/janas/issues/11), 26 Sep 2026 |
| 20.5 | 477.4 | - | AMD Ryzen 7 255 | AMD Radeon 780M Graphics | 31.2 GiB | Linux | v2026.10.12.1 | [#16](https://github.com/prabanta-dev/janas/issues/16), 13 Oct 2026 |
| 17.0 | 58.8 | - | AMD Ryzen 7 5700 | none | 15.8 GiB | Windows | v2026.10.14 | [#36](https://github.com/prabanta-dev/janas/issues/36), 16 Oct 2026 |
| 16.0 | 180.9 | - | Intel Core i5-1135G7 | Intel Iris Xe Graphics | 15.4 GiB | Linux | v2026.10.14 | [#37](https://github.com/prabanta-dev/janas/issues/37), 17 Oct 2026 |
| 5.8 | 69.6 | - | AMD EPYC 9254 | none | 7.7 GiB | Linux | unknown | [#70](https://github.com/prabanta-dev/janas/issues/70), 13 Nov 2026 |

### Qwen3-Coder-30B-A3B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 27.7 | 127.8 | - | Intel Core Ultra 5 125U | Intel Graphics | 30.7 GiB | Linux | 6b8ba9c | [#66](https://github.com/prabanta-dev/janas/issues/66), 12 Nov 2026 |
| 27.6 | 207.7 | - | AMD Ryzen 7 9800X3D | none | 89.5 GiB | Linux | v2026.10.14 | [#24](https://github.com/prabanta-dev/janas/issues/24), 14 Oct 2026 |

### Qwen3-Next-80B-A3B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 16.2 | 61.6 | 23.7 / 25.8 | AMD Ryzen 7 255 | AMD Radeon 780M Graphics | 31.2 GiB | Linux | v2026.10.12.1 | [#16](https://github.com/prabanta-dev/janas/issues/16), 13 Oct 2026 |
| 13.8 | 29.9 | 24.3 / 30.5 | AMD Ryzen 7 7700X | NVIDIA GeForce RTX 3060 | 31.1 GiB | Windows | v2026.10.30.1 | [#49](https://github.com/prabanta-dev/janas/issues/49#issuecomment-6530670350), 30 Oct 2026 |
| 13.7 | 26.3 | 23.5 / 27.2 | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.14 | [#42](https://github.com/prabanta-dev/janas/issues/42), 17 Oct 2026 |
| 12.9 | 47.6 | - | AMD Ryzen 7 4800H | AMD Radeon RX 5600M | 15.0 GiB | Linux | v2026.11.07 | [#63](https://github.com/prabanta-dev/janas/issues/63#issuecomment-6763210439), 10 Nov 2026 |
| 12.1 | 27.8 | 20.3 / 21.6 | AMD Ryzen 7 7700X | NVIDIA GeForce RTX 3060 | 31.1 GiB | Windows | v2026.10.30.1 | [#49](https://github.com/prabanta-dev/janas/issues/49), 30 Oct 2026 |
| 10.8 | 49.0 | 11.9 / 12.7 | AMD Ryzen 9 7900X | NVIDIA GeForce RTX 5070 | 31.1 GiB | Windows | ea67777 | [#47](https://github.com/prabanta-dev/janas/issues/47), 29 Oct 2026 |
| 9.5 | 28.2 | 16.6 / 20.7 | AMD Ryzen 9 5900HX | none | 30.8 GiB | Linux | v2026.10.13 | [#23](https://github.com/prabanta-dev/janas/issues/23), 13 Oct 2026 |
| 8.2 | 22.0 | 13.7 / 14.4 | AMD Ryzen 5 8500G | NVIDIA GeForce RTX 3060 | 30.5 GiB | Linux | v2026.11.03 | [#56](https://github.com/prabanta-dev/janas/issues/56), 3 Nov 2026 |
| 7.3 | 15.7 | 11.0 / 11.2 | Intel Core i5-1135G7 | Intel Iris Xe Graphics | 15.4 GiB | Linux | v2026.10.14 | [#39](https://github.com/prabanta-dev/janas/issues/39), 17 Oct 2026 |

### Qwen3.5-2B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 41.2 | 703.0 | 49.1 / 54.2 | Intel Core Ultra 5 125U | Intel Graphics | 30.7 GiB | Linux | 6b8ba9c | [#68](https://github.com/prabanta-dev/janas/issues/68), 12 Nov 2026 |

### Qwen3.5-9B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 15.2 | 23.7 | 21.0 / 21.2 | AMD Ryzen 7 7840HS | NVIDIA GeForce RTX 4070 Laptop | 31.3 GiB | Windows | v2026.10.14 | [#41](https://github.com/prabanta-dev/janas/issues/41), 17 Oct 2026 |
| 8.9 | 23.8 | - | AMD Ryzen 5 5500 | none | 31.9 GiB | Windows | v2026.10.24 | [#45](https://github.com/prabanta-dev/janas/issues/45), 24 Oct 2026 |

### Qwen3.6-35B-A3B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 35.2 | 81.2 | 46.4 / 46.3 | AMD Ryzen 5 8500G | NVIDIA GeForce RTX 3060 | 30.5 GiB | Linux | v2026.11.03 | [#55](https://github.com/prabanta-dev/janas/issues/55), 3 Nov 2026 |
| 35.2 | 59.4 | 39.8 / 44.5 | AMD Ryzen 5 8500G | NVIDIA GeForce RTX 3060 | 30.5 GiB | Linux | v2026.11.07-1-g6b8ba9c | [#56](https://github.com/prabanta-dev/janas/issues/56#issuecomment-6765113950), 10 Nov 2026 |
| 23.0 | 374.6 | 45.5 / 45.5 | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 61de67d | [#15](https://github.com/prabanta-dev/janas/issues/15), 27 Sep 2026 |
| 22.1 | 338.9 | 43.3 / 44.0 | AMD Ryzen 9 9950X3D | NVIDIA GeForce RTX 5080 | 60.2 GiB | Linux | 61de67d | [#14](https://github.com/prabanta-dev/janas/issues/14), 27 Sep 2026 |
| 13.8 | 244.7 | 23.6 / 21.6 | AMD Ryzen 9 9950X3D | none | 60.2 GiB | Linux | 791b07d | [#11](https://github.com/prabanta-dev/janas/issues/11), 26 Sep 2026 |
| 9.6 | 21.2 | 9.4 / 9.1 | Intel Core i7-13700KF | Tesla V100-PCIE-32GB | 15.4 GiB | Linux | v2026.10.14 | [#27](https://github.com/prabanta-dev/janas/issues/27), 14 Oct 2026 |
| 9.1 | 27.9 | 10.4 / 11.1 | Intel Core i5-1135G7 | Intel Iris Xe Graphics | 15.4 GiB | Linux | v2026.10.18 | [#38](https://github.com/prabanta-dev/janas/issues/38#issuecomment-6249072963), 18 Oct 2026 |
| 7.0 | 26.5 | 7.6 / 7.5 | Intel Core i5-1135G7 | Intel Iris Xe Graphics | 15.4 GiB | Linux | v2026.10.14 | [#38](https://github.com/prabanta-dev/janas/issues/38), 17 Oct 2026 |

### Qwen3.8-27B

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 3.6 | 22.6 | 6.0 / 6.2 | Intel Core Ultra 5 125U | Intel Graphics | 30.7 GiB | Linux | 6b8ba9c | [#71](https://github.com/prabanta-dev/janas/issues/71), 13 Nov 2026 |

### Qwen3.8-Flash-Next

| Reply, tok/s | Prompt, tok/s | With MTP, tok/s | CPU | GPU | RAM | System | Janas | Report |
|---:|---:|---:|---|---|---|---|---|---|
| 9.5 | 38.4 | 11.7 / 14.7 | Intel Core Ultra 5 125U | Intel Graphics | 30.7 GiB | Linux | 6b8ba9c | [#67](https://github.com/prabanta-dev/janas/issues/67), 12 Nov 2026 |

<!-- reports:end -->
