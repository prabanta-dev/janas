/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * win_start.c - what every program of Janas does first on Windows, before
 * main: the console made to show UTF-8, and to take the escape sequences of
 * colours and of the cursor as a Linux terminal does. Linked into the
 * programs only (build.sh), never into the libraries: a program of
 * someone else's that loads them keeps its console as it set it.
 */
#ifdef _WIN32
#include <stdio.h>
#include <windows.h>

__attribute__((constructor)) static void janas_win_start(void)
{
    /* errors and progress written as they come, as C wants of stderr and
       Linux does: without a console Windows's C library kept them in a
       buffer that a server never flushes, and janas-server's log stayed
       empty under Brebus (Windows 11, 17 Nov 2026) */
    setvbuf(stderr, NULL, _IONBF, 0);
    SetConsoleOutputCP(CP_UTF8);
    DWORD std[2] = {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
    for (int i = 0; i < 2; i++) {
        HANDLE h = GetStdHandle(std[i]);
        DWORD m;
        if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &m))
            SetConsoleMode(h, m | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}
#endif
