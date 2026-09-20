# Contributing to Janas

Janas is an experiment in early development: formats, interfaces and the
command line change without notice. Before writing code, it is worth asking in
an issue whether the thing you have in mind is already being changed underneath.

## What helps most

**A measurement from a machine nobody here owns.** Everything in the engine is
decided from what it measures, and it has only ever measured one machine (a
Core Ultra 9 185H with 32 GB of RAM and an NVMe disk, on Debian 13). Run:

```sh
bin/x86_64-linux/janas-bench <model.jns>
```

**with the machine to itself** — close the browser and the editor first. The
engine takes the memory that is free when it starts and reads the disk while it
answers, so anything else running lands in the numbers: on the development
laptop, with 8 GB already taken by other programs, the same model went from
20-25 tokens per second to nearly 18.

Then open a
[hardware report](https://github.com/prabanta-dev/janas/issues/new?template=01-hardware-report.yml):
the form asks for the report it writes, your distribution and kernel
(`uname -a`), your memory (`free -g`), the disk the model sits on, and anything
that looked wrong. The README lists which machines are wanted most: CPUs without
AVX-VNNI, AMD, discrete GPUs, machines with 16 GB or less, slower disks.

**A model that will not convert, or converts and then answers nonsense.** Say
which file (with its link and its SHA-256) and what `gguf2jns` printed
— there is
[a form for that](https://github.com/prabanta-dev/janas/issues/new?template=03-a-model-will-not-work.yml)
too.

## Code

- Build and test before sending anything:

  ```sh
  ./build.sh release test
  ./build.sh asan test        # and tsan, if you touched anything concurrent
  ```

  `./build.sh --help` says what each variant catches and what the actions do.

- The code is C11 with no dependencies beyond the C library. Vulkan, when used,
  is loaded at run time: the engine must build and run without it.
- Format with `clang-format -i` (the `.clang-format` in the root). Warnings are
  errors: `-Wall -Wextra -Werror`.
- A change that touches arithmetic must keep the results it promises: kernels
  agree bit for bit with the scalar reference, and a pass over a block of tokens
  agrees bit for bit with the same tokens one at a time. The tests check this;
  keep them passing.
- Numbers in a commit message are welcome, and should say on what they were
  measured.

## Copyright and sign-off

Janas is licensed under the **GNU General Public License, version 3 or later**
(see [LICENSE](LICENSE)). Contributions are taken under the same licence.

Every commit must carry a sign-off line:

```
Signed-off-by: Your Name <your@email>
```

which `git commit -s` adds for you. It means what the
[Developer Certificate of Origin](https://developercertificate.org/) says: that
you wrote the change, or have the right to submit it under this licence.

Please keep contributions to code you own. Code copied from elsewhere, however
small, has to be declared in the pull request, with where it came from and under
what licence.

## Models

The engine reads models; it does not carry them. Anything about redistributing
converted weights is in [MODELS.md](MODELS.md) — in short, the weights keep
their own licence, which is not this one.
