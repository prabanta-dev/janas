/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* WebView2.h includes "EventToken.h", as Windows's SDK names it; MinGW's
   is eventtoken.h. Including that from here finds this very file again on
   a file system that does not tell the two apart (NTFS: a build on
   Windows itself, issue #75), so the one type WebView2.h takes from it is
   defined here, under MinGW's own guard: whichever of the two comes
   first, the other adds nothing. INT64 is Windows's, already there by
   the headers WebView2.h includes before. */
#ifndef __eventtoken_h__
#define __eventtoken_h__
typedef struct EventRegistrationToken {
    INT64 value;
} EventRegistrationToken;
#endif
