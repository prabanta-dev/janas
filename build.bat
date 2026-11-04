@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 Maurizio Cammalleri
rem
rem build.bat - Janas built on Windows itself: build.sh run in MSYS2's
rem MINGW64 shell (https://www.msys2.org, installed in C:\msys64, or where
rem MSYS2_ROOT says). The arguments are build.sh's:
rem   build.bat                   a release build into bin\x86_64-windows\
rem   build.bat release test      and its tests
rem   build.bat setup             first, once: the compiler, git, glslang
rem                               and the Vulkan headers, from MSYS2
setlocal
cd /d "%~dp0"
set "M=%MSYS2_ROOT%"
if "%M%"=="" set "M=C:\msys64"
if not exist "%M%\usr\bin\bash.exe" (
    echo build.bat: MSYS2 not found in %M%: install it from https://www.msys2.org
    echo or set MSYS2_ROOT to where it is
    exit /b 2
)
set MSYSTEM=MINGW64
set CHERE_INVOKING=1
if /i "%~1"=="setup" (
    "%M%\usr\bin\bash.exe" -lc "pacman -S --needed --noconfirm git mingw-w64-x86_64-gcc mingw-w64-x86_64-glslang mingw-w64-x86_64-vulkan-headers"
    exit /b %ERRORLEVEL%
)
"%M%\usr\bin\bash.exe" -lc "./build.sh %*"
exit /b %ERRORLEVEL%
