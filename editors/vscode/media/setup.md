# Set up Janas

Janas-LLM runs the model on this machine: the CPU, and the GPU or the NPU where there is one. A mixture of experts larger than the memory still runs, its experts read from the disk as they are needed.

**Set up Janas** does the whole way:

1. Janas, from github.com/prabanta-dev/janas, its package checked against the SHA-256 published beside it, installed in `~/.local/share/janas` (Linux) or `%LOCALAPPDATA%\janas` (Windows).
2. A model for this machine's memory.
3. An extension to chat and complete code with, if there is none.
4. That extension set up to use Janas, and Janas's servers started.

Nothing you write leaves the machine.
