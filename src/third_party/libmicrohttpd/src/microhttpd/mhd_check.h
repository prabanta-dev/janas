/*
  This file is part of libmicrohttpd
  Copyright (C) 2026 Christian Grothoff

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library.
  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file microhttpd/mhd_check.h
 * @brief  macros for MHD_CHECK_(), always-compiled memory-safety invariants
 * @author Christian Grothoff
 *
 * mhd_assert() is compiled out in release builds (NDEBUG), which is the right
 * thing for the several hundred internal state assertions of this library but
 * the wrong thing for the handful of them that stand between attacker-supplied
 * data and a memmove()/memcpy()/array index.  For those, an assertion that is
 * absent in the shipped build is not a safety net at all.
 *
 * MHD_CHECK_() is the always-compiled companion for exactly that subset.  It
 * does NOT depend on NDEBUG, _DEBUG or --enable-asserts: the test is one
 * predictable, well-predicted branch that is present in every build.
 *
 * RULES FOR USING IT
 * ------------------
 * 1. Only invariants whose violation would mean memory-unsafe behaviour -
 *    pointer/offset arithmetic, buffer bounds, length subtractions that could
 *    underflow, indices into fixed-size arrays.  Ordinary state assertions
 *    stay mhd_assert().
 * 2. Check the *precondition*, not the symptom.  An assertion that compares
 *    two pointers after the bad one has already been derived is decoration:
 *    it happily passes on a NULL-derived pointer.  Assert what has to be true
 *    for the following arithmetic to be defined.  See commit 29eaa56b.
 * 3. A failing check must FAIL THE CONNECTION - close it, or reply 4xx/5xx -
 *    and must not continue and must not panic.  MHD_PANIC() aborts the whole
 *    daemon, which turns a bounded per-connection problem into a global
 *    denial of service; that is the opposite of what this is for.
 * 4. Deliberately no mhd_assert() inside: a check must behave identically in
 *    debug and in release builds, otherwise the debug build stops exercising
 *    the recovery path that the release build relies on.
 *
 * WHERE THE MACROS COME FROM
 * --------------------------
 * The connection-level macros below expand to the failure exits that
 * connection.c already uses - connection_close_error() /
 * CONNECTION_CLOSE_ERROR() and transmit_error_response_static() - rather than
 * introducing a parallel mechanism.  They are therefore usable only from
 * connection.c and only after those helpers have been defined; this header
 * intentionally does not include connection.c's internals, the macros are
 * expanded at the point of use.
 *
 * Translation units that have a `struct MHD_Daemon *` at hand (connection.c,
 * digestauth.c, postprocessor.c, ...) get a log line through the usual
 * MHD_DLOG()/HAVE_MESSAGES machinery; they must include "internal.h" before
 * this header.  Translation units without a daemon (memorypool.c, mhd_str.c)
 * use the MHD_CHECK_RET_() form, which is silent.
 */

#ifndef MHD_CHECK_H
#define MHD_CHECK_H 1

#include "mhd_options.h"

/**
 * Branch hint: an invariant violation is by construction the unlikely case.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define MHD_CHECK_FAILED_(expr) (__builtin_expect (! (expr), 0))
#else  /* ! __GNUC__ */
#  define MHD_CHECK_FAILED_(expr) (! (expr))
#endif /* ! __GNUC__ */

/**
 * Report a violated invariant through the daemon's log.
 *
 * @param daemon the daemon to log to
 * @param expr_str the stringified invariant that was violated
 */
#ifdef HAVE_MESSAGES
#  define MHD_CHECK_LOG_(daemon,expr_str) \
        MHD_DLOG (daemon, \
                  _ ("Internal invariant violated at %s:%u: (%s). " \
                     "Failing the connection.\n"), \
                  __FILE__, \
                  (unsigned int) __LINE__, \
                  expr_str)
#else  /* ! HAVE_MESSAGES */
#  define MHD_CHECK_LOG_(daemon,expr_str) ((void) 0)
#endif /* ! HAVE_MESSAGES */


/**
 * The generic always-compiled invariant check.
 *
 * @param daemon the daemon to log to
 * @param expr the invariant, must evaluate to true
 * @param fail_stmt a single statement to execute when @a expr is false; it
 *                  must transfer control out of the current function
 *                  (return, goto or break), because execution must not
 *                  continue past a violated memory-safety invariant.
 *                  When the failure action needs several statements, spell
 *                  the check out instead - "if (MHD_CHECK_FAILED_ (expr)) {
 *                  MHD_CHECK_LOG_ (daemon, "expr"); ... }" - because a
 *                  braced block passed as a macro argument is re-indented
 *                  by contrib/uncrustify.cfg
 */
#define MHD_CHECK_(daemon,expr,fail_stmt) \
        do { \
          if (MHD_CHECK_FAILED_ (expr)) \
          { \
            MHD_CHECK_LOG_ (daemon, #expr); \
            fail_stmt; \
          } \
        } while (0)


/**
 * Always-compiled invariant check for translation units that have no daemon
 * pointer available for logging (memorypool.c, mhd_str.c).  Silent; the
 * caller is expected to turn the returned failure value into a log message
 * and a failed connection.
 *
 * @param expr the invariant, must evaluate to true
 * @param retval the value to return when @a expr is false
 */
#define MHD_CHECK_RET_(expr,retval) \
        do { \
          if (MHD_CHECK_FAILED_ (expr)) \
            return retval; \
        } while (0)


/**
 * Always-compiled invariant check that closes the connection with an error
 * and returns @a retval.  For use in connection.c only, and only after
 * connection_close_error() has been defined.
 *
 * @param c the connection to fail
 * @param expr the invariant, must evaluate to true
 * @param retval the value to return when @a expr is false
 */
#define MHD_CHECK_CONN_CLOSE_RET_(c,expr,retval) \
        MHD_CHECK_ ((c)->daemon, \
                    expr, \
                    { \
                      connection_close_error ((c), \
                                              NULL); \
                      return retval; \
                    })


/**
 * Always-compiled invariant check that closes the connection with an error
 * and returns from a void function.  For use in connection.c only, and only
 * after connection_close_error() has been defined.
 *
 * @param c the connection to fail
 * @param expr the invariant, must evaluate to true
 */
#define MHD_CHECK_CONN_CLOSE_RET_VOID_(c,expr) \
        MHD_CHECK_ ((c)->daemon, \
                    expr, \
                    { \
                      connection_close_error ((c), \
                                              NULL); \
                      return; \
                    })


/**
 * Always-compiled invariant check that queues a static error reply for the
 * connection and returns @a retval.  For use in connection.c only, and only
 * after transmit_error_response_static() has been defined.
 *
 * @param c the connection to fail
 * @param expr the invariant, must evaluate to true
 * @param code the HTTP status code to reply with
 * @param msg the static message to reply with
 * @param retval the value to return when @a expr is false
 */
#define MHD_CHECK_CONN_REPLY_RET_(c,expr,code,msg,retval) \
        MHD_CHECK_ ((c)->daemon, \
                    expr, \
                    { \
                      transmit_error_response_static ((c), \
                                                      (code), \
                                                      msg); \
                      return retval; \
                    })

#endif /* ! MHD_CHECK_H */
