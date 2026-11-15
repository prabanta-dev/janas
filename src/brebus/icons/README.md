# Brebus's icons

The Prabanta logo (`icon.svg`, the original), drawn for each system's rules:

| File | For | |
|---|---|---|
| `icon16.png` … `icon512.png` | Linux (freedesktop `hicolor`: 16, 22, 24, 32, 48, 64, 128, 256, 512) | a 1/16 margin from 32 px up, as Breeze's and Adwaita's application icons; up to 24 px the strokes 1.6 times as thick and the dots larger, so that they do not blur |
| `icon.svg` | Linux (`hicolor/scalable`) | the logo as it was drawn |
| `brebus.ico` | Windows | 16, 20, 24, 32, 40, 48, 64, 96, 128 and 256 px, each drawn on its own, full size |
| `brebus.icns` | macOS | 16 to 1024 px (with the @2x ones), the logo at 824 of 1024 px with its corners rounded at 22.37%, Apple's template since macOS 11 |

Brebus carries the PNGs and the SVG (`icons.c`): the page's logo, the window's icon, and those written in `~/.local/share/icons/hicolor` with `brebus.desktop` when the window opens or with `brebus --install-desktop`. The `.ico` and `.icns` are for the Windows and macOS builds of Brebus, which are to come.

They were made from `icon.svg` with ImageMagick (the SVG drawn at 2048 px) and Pillow (each size from it with Lanczos, the ICO and ICNS written).
