/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* sys/resource.h for Windows (src/common/win/): the process's times and
   its peak working set as getrusage's; the process's priority as
   setpriority's, a nice above 0 its class below normal, above 10 idle
   (win_posix.c). */
#ifndef JANAS_WIN_SYS_RESOURCE_H
#define JANAS_WIN_SYS_RESOURCE_H
#include <sys/time.h>
#define RUSAGE_SELF 0
struct rusage {
    struct timeval ru_utime, ru_stime;
    long ru_maxrss; /* KiB, as Linux's */
};
int getrusage(int who, struct rusage *u);
#define PRIO_PROCESS 0
int setpriority(int which, int who, int nice);
#endif
