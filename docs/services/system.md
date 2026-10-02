# janas-system

This computer for `janas-chat` or any other MCP client (see [the services](README.md)): how it is, its disks and what fills a directory, the programs running, its errors, its updates, its network. It only reads, and none of its tools reaches the network: what it reads leaves the computer only in its answers.

## The tools

- **`system_status`** - the name of the computer and its system, how long it has been on; the processor (its share in use over half a second, and how much of it the kernel's; the load where the system tells it); the memory and the swap; on Linux the time the programs waited for the processor, the memory and the disks (pressure), when it is high; every disk's space, those 90% full told; the battery (its charge, its state, the time left or to full, its capacity against what it was new); the temperatures, one a device (the processor and its hottest core, a disk, the memory, the Wi-Fi card, the motherboard), those within 10 °C of their critical limit told.
- **`system_disks`** - the disks: each filesystem once, its device, its type, its space. With a path (`path`, `~` for the home), what fills that directory, biggest first: each entry's size, the directories' summed over what they hold, with their files counted.
- **`system_processes`** - the programs that use the most processor (`by`: `cpu`) or memory (`memory`), or those whose name has a text (`name`: is a program running?): each with its process number, its user, its share of the whole processor over half a second, its memory, how long it has been running. A program's name only, never its arguments.
- **`system_errors`** - the errors of the system's logs since the computer started or in the last hours (`hours`), the same message of the same program counted once with its latest time, and the services that failed.
- **`system_updates`** - the updates of the installed packages, those for security counted, by the list the package manager last fetched (and when that was), with the command that fetches it again.
- **`system_network`** - the network interfaces, their state and addresses, the speed of a cable, the Wi-Fi signal, what each has received and sent; the gateway and the name servers.

## Where it reads

| | Linux | Windows |
|---|---|---|
| memory, processor | `/proc` | `GlobalMemoryStatusEx`, `GetSystemTimes` |
| disks | `/proc/self/mountinfo`, `statvfs` | the drives, `GetDiskFreeSpaceEx` |
| battery | `/sys/class/power_supply` | `GetSystemPowerStatus` (its wear is not told) |
| temperatures | `/sys/class/hwmon` | not told: they need WMI and an administrator |
| processes | `/proc/<pid>` | Tool Help, `GetProcessTimes` |
| errors | `journalctl` (priority error and above), `systemctl --failed` (the system's and the user's) | the event log (System and Application, levels critical and error), the services set to start by themselves and stopped with an error |
| updates | `apt list --upgradable`, `dnf check-update --cacheonly`, `pacman -Qu`: what they already know, nothing fetched | not read: they are Windows Update's (the answer says where to look) |
| network | `getifaddrs`, `/sys/class/net`, `/proc/net/wireless`, `/proc/net/route`, `/etc/resolv.conf` | IP Helper, the WLAN API |

- **Within limits.** What fills a directory is read for 8 seconds at most, 64 levels deep at most, never across to another filesystem nor through a symbolic link (on Windows, a junction or a mounted volume); when the time runs out the answer says the sizes are at least those. On Linux a size is the space the files take on the disk; on Windows, their lengths. A file of more names than one (hard links) is counted at each.
- **The logs as you may read them.** Without the groups `adm` or `systemd-journal`, `journalctl` shows only your own programs' and session's messages: the answer says so, and asks for nothing more. The latest are read first; when there are more than 2,000, or more than a megabyte of them, the answer says the count is at least that. A message is its first line (a core dump's modules and stack are for the log's own reader), and the same text but for its numbers (a process's number, a time) is counted once.
- **Nothing that needs root**, nothing run through a shell: the programs it starts run with their arguments as they are, without a terminal, with a time limit (as `janas-git`'s).
- **Nothing changed.** To stop a program or install the updates, the model tells you how.

## The answers

Every answer is data with a layout (see [the services](README.md)): `janas-chat` shows you the whole of it in your language - every disk, process, error, package - and gives the model only a brief, from which it answers in a line or two; for the status, it is told to say only what is wrong (a disk nearly full, memory short, a temperature near its limit, a worn battery), or that all is well. Sizes, percentages and times are worked out by the service, so that the model does no sums.

## Running it

```sh
janas-system
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add system -- /path/to/janas-system
```

```json
{"mcpServers": {"system": {"command": "/path/to/janas-system"}}}
```

A client that sends its model's requests to a server elsewhere sends there the briefs too: the names of the programs running, the paths in your home, the lines of the logs.

## Windows

Janas is not built for Windows yet. `janas-system`'s Windows part is written for the Win32 API, and has been compiled and linked with MinGW-w64; it has not been run on Windows.

## What has been run

2 October 2026, on the development laptop (Debian 13), over the protocol, in the release build: the status (with the machine busy with other work, the load at 17 to 25), the disks, what fills the home (the time ran out after 8 seconds, at 55.6 GB), the processes by processor, by memory and by name, the errors since the start and in the last 2 hours, the updates (96, 95 for security, by apt's list), the network, a path that does not exist. The parsing of Linux's texts and the layouts are checked by `tests/test_system_parse.c`, on texts of that machine; those of dnf and pacman are written from their documentation, as no such system was at hand.

In `janas-chat` with Qwen3.6-35B-A3B, the same night, in Italian, on a pipe: "Come sta il computer?" (all is well, it said), "Cosa occupa spazio nella home?", "Quali programmi usano più memoria?", "Gira Firefox?", "Ci sono errori nelle ultime ore?" (it took the last 24 hours), "Ci sono aggiornamenti?": each answered from its tool, the lists shown in Italian. The status's layout came back from the model untranslated, and was refused (English was shown); `janas-chat` now asks once more when that happens, and run again, the status came in Italian.

The Windows part, under Wine 10 in a prefix of its own: the status, the processes, the errors of the last 24 hours, the network and what fills `C:\windows` came back without errors; Wine gives no event log and no Wi-Fi signal of its own, so those were not tried.
