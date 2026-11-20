# WebView2's header

`WebView2.h` and `LICENSE.txt` of Microsoft's WebView2 SDK, the NuGet package `Microsoft.Web.WebView2` 1.0.4314.35 (published 23 October 2026, SHA-256 of the `.nupkg` `90778deb1b77e36292828b4b5833c9255c27533f40e0ddd55891ae718944edc4`), unchanged, under its BSD-style licence (`LICENSE.txt`). Brebus's window on Windows (`src/brebus/window_win.c`) is written against it.

`WebView2Loader.dll` of the same package (`build/native/x64/`) goes beside `brebus.exe` in the Windows package; it is not kept here. `EventToken.h` is Janas's: it defines the one type the header takes from it, `EventRegistrationToken`, under the guard of MinGW's `eventtoken.h`, so that a build on Windows itself, whose file system does not tell `EventToken.h` and `eventtoken.h` apart, does not include it in a loop.
