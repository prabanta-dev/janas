# Releases and the develop branch

Janas has two branches on GitHub, and they move at different times.

- **`main` is the last release.** It changes only when a release is made, and each release has a tag (`v2026.11.13`) and its packages for Linux and Windows on the [releases page](https://github.com/prabanta-dev/janas/releases). If you want something that has been through a release's checks, use this one.
- **`develop` is what comes next.** A fix or a new feature goes there as soon as it is done and its tests pass, usually long before the next release. When an issue you opened is answered with "it's in develop", this is the branch to build.

From time to time `develop` goes onto `main` and becomes a release, with new packages. Until then, the only way to try what is in `develop` is to build it, which takes a few minutes and is described below.

## Getting develop

With git, the first time:

```sh
git clone -b develop https://github.com/prabanta-dev/janas.git
cd janas
```

and afterwards, to bring it up to date:

```sh
git pull
```

If you cloned `main` before, `git checkout develop && git pull` moves you to develop, and `git checkout main` takes you back.

Without git, as a ZIP file: <https://github.com/prabanta-dev/janas/archive/refs/heads/develop.zip>. It unpacks into a folder named `janas-develop`. To update, download it again.

## Preparing to build

**Linux** (Debian, Ubuntu): the compiler, and for the GPU Vulkan's headers and the shader compiler:

```sh
sudo apt install build-essential git
sudo apt install libvulkan-dev glslang-tools vulkan-tools   # optional, for the GPU
```

Fedora, Arch, the glslang version the GPU needs (15.2 or later: Ubuntu 24.04's is older) and Intel's NPU driver are in [INSTALL.md](../INSTALL.md#building).

**Windows** 10 (1903 or later) and 11: install [MSYS2](https://www.msys2.org) in `C:\msys64` (or anywhere, with `MSYS2_ROOT` set to its folder). Then, once, from a command prompt in Janas's folder:

```bat
build.bat setup
```

This installs the compiler, git, glslang and the Vulkan headers from MSYS2.

## Building

**Linux**, in Janas's folder:

```sh
./build.sh                  # everything into bin/x86_64-linux/
./build.sh release test     # the same, then the tests
```

**Windows**, from a command prompt in Janas's folder:

```bat
build.bat                   :: everything into bin\x86_64-windows\
build.bat release test      :: the same, then the tests
```

The programs are then in `bin/x86_64-linux/` (or `bin\x86_64-windows\`): `janas-chat`, `janas-bench`, `janas-get`, `janas-server` and the others, used as in the release packages. `janas-bench`'s first line says which commit it was built from (`janas-bench 1bc6615, ...`), which is worth quoting in a report; built from a ZIP, which carries no history, it says `unknown`, and the date you downloaded it does the same job. Building again after a `git pull` (or a new ZIP) recompiles only what changed.

When the build or a test fails, an issue with its output helps: [INSTALL.md](../INSTALL.md#troubleshooting) has the usual causes.
