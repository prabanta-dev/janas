/* MHD_config.h.  Generated from MHD_config.h.in by configure.  */
/* MHD_config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define to 1 if libmicrohttpd is compiled with Basic Auth support. */
/* #undef BAUTH_SUPPORT */

/* Define to 1 if libmicrohttpd is compiled with HTTP cookie parsing support.
   */
/* #undef COOKIE_SUPPORT */

/* This is a Cygwin system */
/* #undef CYGWIN */

/* Define to 1 if libmicrohttpd is compiled with Digest Auth support. */
/* #undef DAUTH_SUPPORT */

/* Define to 1 to enable epoll support */
#define EPOLL_SUPPORT 1

/* This is a FreeBSD system */
/* #undef FREEBSD */

/* Define to '1' if '__attribute__((no_sanitize("address")))' works for
   pointers compare */
/* #undef FUNC_ATTR_NOSANITIZE_WORKS */

/* Define to '1' if '__attribute__((no_sanitize("pointer-compare")))' works */
/* #undef FUNC_ATTR_PTRCOMPARE_WORKS */

/* Define to '1' if '__attribute__((no_sanitize("pointer-subtract")))' works
   */
/* #undef FUNC_ATTR_PTRSUBTRACT_WORKS */

/* Define to '1' if cast to 'uintptr_t' works for safely processing
   user-poisoned pointer */
/* #undef FUNC_PTRCOMPARE_CAST_WORKAROUND_WORKS */

/* Define to `1' if host machine runs on GNU Hurd. */
/* #undef GNU_HURD */

/* Define to 1 if your system allow overriding the value of FD_SETSIZE macro
   */
/* #undef HAS_FD_SETSIZE_OVERRIDABLE */

/* Define to 1 if your socket type is signed */
#define HAS_SIGNED_SOCKET 1

/* Define to 1 if you have the 'accept4' function. */
#define HAVE_ACCEPT4 1

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define if you have usable assert() and assert.h */
/* #undef HAVE_ASSERT */

/* Define to 1 if you have the boolean type that takes only 'true' and
   'false'. */
#define HAVE_BUILTIN_TYPE_BOOL 1

/* Define to 1 if you have the `gmtime_s' function in C11 form. */
/* #undef HAVE_C11_GMTIME_S */

/* Define to 1 if you have the usable `calloc' function. */
#define HAVE_CALLOC 1

/* Define to '1' if you have clock_gettime() function */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if you have the 'clock_get_time' function. */
/* #undef HAVE_CLOCK_GET_TIME */

/* Define to 1 if you have the 'cpuset_getaffinity' function. */
/* #undef HAVE_CPUSET_GETAFFINITY */

/* Define to 1 if you have the 'CPU_COUNT' function. */
/* #undef HAVE_CPU_COUNT */

/* Define to 1 if you have the 'CPU_COUNT_S' function. */
/* #undef HAVE_CPU_COUNT_S */

/* Define to 1 if your compiler supports 'alignof()' */
#define HAVE_C_ALIGNOF 1

/* Define to 1 if C supports variable-length arrays. */
#define HAVE_C_VARARRAYS 1

/* Define to 1 if you have Darwin-style sendfile(2). */
/* #undef HAVE_DARWIN_SENDFILE */

/* Define to 1 if you have the declaration of 'CPU_SETSIZE', and to 0 if you
   don't. */
/* #undef HAVE_DECL_CPU_SETSIZE */

/* Define to 1 if you have the declaration of 'CTL_HW', and to 0 if you don't.
   */
/* #undef HAVE_DECL_CTL_HW */

/* Define to 1 if you have the declaration of 'CTL_NET', and to 0 if you
   don't. */
/* #undef HAVE_DECL_CTL_NET */

/* Define to 1 if you have the declaration of 'HW_AVAILCPU', and to 0 if you
   don't. */
/* #undef HAVE_DECL_HW_AVAILCPU */

/* Define to 1 if you have the declaration of 'HW_NCPU', and to 0 if you
   don't. */
/* #undef HAVE_DECL_HW_NCPU */

/* Define to 1 if you have the declaration of 'HW_NCPUONLINE', and to 0 if you
   don't. */
/* #undef HAVE_DECL_HW_NCPUONLINE */

/* Define to 1 if you have the declaration of 'ICMPCTL_ICMPLIM', and to 0 if
   you don't. */
/* #undef HAVE_DECL_ICMPCTL_ICMPLIM */

/* Define to 1 if you have the declaration of 'IPPROTO_ICMP', and to 0 if you
   don't. */
/* #undef HAVE_DECL_IPPROTO_ICMP */

/* Define to 1 if you have the declaration of 'PF_INET', and to 0 if you
   don't. */
/* #undef HAVE_DECL_PF_INET */

/* Define to 1 if you have the declaration of '_SC_CRAY_NCPU', and to 0 if you
   don't. */
/* #undef HAVE_DECL__SC_CRAY_NCPU */

/* Define to 1 if you have the declaration of '_SC_NPROCESSORS_CONF', and to 0
   if you don't. */
/* #undef HAVE_DECL__SC_NPROCESSORS_CONF */

/* Define to 1 if you have the declaration of '_SC_NPROCESSORS_ONLN', and to 0
   if you don't. */
/* #undef HAVE_DECL__SC_NPROCESSORS_ONLN */

/* Define to 1 if you have the declaration of '_SC_NPROC_ONLN', and to 0 if
   you don't. */
/* #undef HAVE_DECL__SC_NPROC_ONLN */

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <endian.h> header file. */
#define HAVE_ENDIAN_H 1

/* Define to 1 if you have the 'epoll_create1' function. */
#define HAVE_EPOLL_CREATE1 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <features.h> header file. */
/* #undef HAVE_FEATURES_H */

/* Define to 1 if you have the usable `fork' function. */
#define HAVE_FORK 1

/* Define to 1 if you have FreeBSD-style sendfile(2). */
/* #undef HAVE_FREEBSD_SENDFILE */

/* Define to 1 if fseeko (and ftello) are declared in stdio.h. */
#define HAVE_FSEEKO 1

/* Define to 1 if you have the <gcrypt.h> header file. */
/* #undef HAVE_GCRYPT_H */

/* Define to 1 if you have the 'gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if you have the 'getpid' function. */
/* #undef HAVE_GETPID */

/* Define to 1 if you have the 'gettimeofday' function. */
#define HAVE_GETTIMEOFDAY 1

/* Define to 1 if you have the 'gmtime_r' function. */
#define HAVE_GMTIME_R 1

/* Define to 1 if you have the 'gnutls_check_version' function. */
/* #undef HAVE_GNUTLS_CHECK_VERSION */

/* Define to 1 if you have the <gnutls/gnutls.h> header file. */
/* #undef HAVE_GNUTLS_GNUTLS_H */

/* Define to '1' if you have IPv6 headers */
#define HAVE_INET6 1

/* Define to 1 if you have the <inetLib.h> header file. */
/* #undef HAVE_INETLIB_H */

/* Define to 1 if your C compiler supports inline functions. */
#define HAVE_INLINE_FUNCS 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <jansson.h> header file. */
/* #undef HAVE_JANSSON_H */

/* Define to 1 if you have a functional curl library. */
/* #undef HAVE_LIBCURL */

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if you have linux-style sendfile(2). */
#define HAVE_LINUX_SENDFILE 1

/* can use shutdown on listen sockets */
#define HAVE_LISTEN_SHUTDOWN 1

/* Define to 1 if you have the 'lseek64' function. */
#define HAVE_LSEEK64 1

/* Define to 1 if you have the <machine/endian.h> header file. */
/* #undef HAVE_MACHINE_ENDIAN_H */

/* Define to 1 if you have the <machine/param.h> header file. */
/* #undef HAVE_MACHINE_PARAM_H */

/* Define to 1 if you have the 'magic_open' function. */
/* #undef HAVE_MAGIC_OPEN */

/* Define to 1 if you have the 'memmem' function. */
#define HAVE_MEMMEM 1

/* Define to 1 to enable support for text messages. */
#define HAVE_MESSAGES 1

/* Define to 1 if you have the 'nanosleep' function. */
#define HAVE_NANOSLEEP 1

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/icmp_var.h> header file. */
/* #undef HAVE_NETINET_ICMP_VAR_H */

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/in_systm.h> header file. */
#define HAVE_NETINET_IN_SYSTM_H 1

/* Define to 1 if you have the <netinet/ip.h> header file. */
#define HAVE_NETINET_IP_H 1

/* Define to 1 if you have the <netinet/ip_icmp.h> header file. */
#define HAVE_NETINET_IP_ICMP_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define if you have usable pipe2(2) function */
/* #undef HAVE_PIPE2_FUNC */

/* Define to 1 if you have the 'poll' function. */
#define HAVE_POLL 1

/* Define to 1 if you have the <poll.h> header file. */
#define HAVE_POLL_H 1

/* Define to 1 if libmicrohttpd is compiled with postprocessor support. */
/* #undef HAVE_POSTPROCESSOR */

/* Define to 1 if you have the 'pread' function. */
#define HAVE_PREAD 1

/* Define to 1 if you have the 'pread64' function. */
#define HAVE_PREAD64 1

/* Define to 1 if you have the 'pstat_getdynamic' function. */
/* #undef HAVE_PSTAT_GETDYNAMIC */

/* Define if you have IBM i form (or Solaris form) of
   pthread_attr_setname_np(3) function. */
/* #undef HAVE_PTHREAD_ATTR_SETNAME_NP_IBMI */

/* Define if you have NetBSD form (or OSF1 form) of pthread_attr_setname_np(3)
   function. */
/* #undef HAVE_PTHREAD_ATTR_SETNAME_NP_NETBSD */

/* Define to 1 if you have the <pthread.h> header file. */
#define HAVE_PTHREAD_H 1

/* Define to 1 if you have the <pthread_np.h> header file. */
/* #undef HAVE_PTHREAD_NP_H */

/* Have PTHREAD_PRIO_INHERIT. */
#define HAVE_PTHREAD_PRIO_INHERIT 1

/* Define if you have Darwin form of pthread_setname_np(3) function. */
/* #undef HAVE_PTHREAD_SETNAME_NP_DARWIN */

/* Define if you have GNU/Linux form of pthread_setname_np(3) function. */
#define HAVE_PTHREAD_SETNAME_NP_GNU 1

/* Define if you have NetBSD form (or OSF1 form) of pthread_setname_np(3)
   function. */
/* #undef HAVE_PTHREAD_SETNAME_NP_NETBSD */

/* Define if you have FreeBSD form of pthread_set_name_np(3) function. */
/* #undef HAVE_PTHREAD_SET_NAME_NP_FREEBSD */

/* Define to 1 if you have the pthread_sigmask(3) function. */
#define HAVE_PTHREAD_SIGMASK 1

/* Define to 1 if you have the 'rand' function. */
/* #undef HAVE_RAND */

/* Define to 1 if you have the 'random' function. */
#define HAVE_RANDOM 1

/* Define to 1 if you have the <sanitizer/asan_interface.h> header file. */
/* #undef HAVE_SANITIZER_ASAN_INTERFACE_H */

/* Define to 1 if you have the 'sched_getaffinity' function. */
/* #undef HAVE_SCHED_GETAFFINITY */

/* Define to 1 if you have the 'sched_getaffinity_np' function. */
/* #undef HAVE_SCHED_GETAFFINITY_NP */

/* Define to 1 if you have the <sched.h> header file. */
/* #undef HAVE_SCHED_H */

/* Define to 1 if you have the <sdkddkver.h> header file. */
/* #undef HAVE_SDKDDKVER_H */

/* Define to 1 if you have the 'sendfile64' function. */
#define HAVE_SENDFILE64 1

/* Define to '1' if your have sendmsg() function */
#define HAVE_SENDMSG 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the 'snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define to 1 if you have the <sockLib.h> header file. */
/* #undef HAVE_SOCKLIB_H */

/* SOCK_NONBLOCK is defined in a socket header */
#define HAVE_SOCK_NONBLOCK 1

/* Define to 1 if you have Solaris-style sendfile(3). */
/* #undef HAVE_SOLARIS_SENDFILE */

/* Define to 1 if you have the <stdalign.h> header file. */
#define HAVE_STDALIGN_H 1

/* Define to 1 if you have the <stdbool.h> header file and <stdbool.h> is
   required to use 'bool' type. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if 'sin6_len' is a member of 'struct sockaddr_in6'. */
/* #undef HAVE_STRUCT_SOCKADDR_IN6_SIN6_LEN */

/* Define to 1 if 'sin_len' is a member of 'struct sockaddr_in'. */
/* #undef HAVE_STRUCT_SOCKADDR_IN_SIN_LEN */

/* Define to 1 if 'sa_len' is a member of 'struct sockaddr'. */
/* #undef HAVE_STRUCT_SOCKADDR_SA_LEN */

/* Define to 1 if 'ss_len' is a member of 'struct sockaddr_storage'. */
/* #undef HAVE_STRUCT_SOCKADDR_STORAGE_SS_LEN */

/* Define to 1 if you have the 'sysconf' function. */
#define HAVE_SYSCONF 1

/* Define to 1 if you have the 'sysctl' function. */
/* #undef HAVE_SYSCTL */

/* Define to 1 if you have the 'sysctlbyname' function. */
/* #undef HAVE_SYSCTLBYNAME */

/* Define to 1 if you have the <sys/byteorder.h> header file. */
/* #undef HAVE_SYS_BYTEORDER_H */

/* Define to 1 if you have the <sys/cpuset.h> header file. */
/* #undef HAVE_SYS_CPUSET_H */

/* Define to 1 if you have the <sys/endian.h> header file. */
/* #undef HAVE_SYS_ENDIAN_H */

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/isa_defs.h> header file. */
/* #undef HAVE_SYS_ISA_DEFS_H */

/* Define to 1 if you have the <sys/machine.h> header file. */
/* #undef HAVE_SYS_MACHINE_H */

/* Define to 1 if you have the <sys/mman.h> header file. */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/msg.h> header file. */
#define HAVE_SYS_MSG_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/pstat.h> header file. */
/* #undef HAVE_SYS_PSTAT_H */

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/sysctl.h> header file. */
/* #undef HAVE_SYS_SYSCTL_H */

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/_cpuset.h> header file. */
/* #undef HAVE_SYS__CPUSET_H */

/* Define to 1 if you have the 'timespec_get' function. */
#define HAVE_TIMESPEC_GET 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the 'usleep' function. */
#define HAVE_USLEEP 1

/* Define to 1 if you have the 'vxCpuEnabledGet' function. */
/* #undef HAVE_VXCPUENABLEDGET */

/* Define to 1 if you have the <vxCpuLib.h> header file. */
/* #undef HAVE_VXCPULIB_H */

/* Define to 1 if you have the `gmtime_s' function in W32 form. */
/* #undef HAVE_W32_GMTIME_S */

/* Define to 1 if you have the usable `waitpid' function. */
#define HAVE_WAITPID 1

/* Define to 1 if you have the <windows.h> header file. */
/* #undef HAVE_WINDOWS_H */

/* Define to 1 if you have the <winsock2.h> header file. */
/* #undef HAVE_WINSOCK2_H */

/* Define to 1 if you have the 'writev' function. */
#define HAVE_WRITEV 1

/* Define to 1 if you have the <ws2tcpip.h> header file. */
/* #undef HAVE_WS2TCPIP_H */

/* Define to 1 if you have the 'WSAPoll' function. */
/* #undef HAVE_WSAPOLL */

/* Define to 1 if you have the <zlib.h> header file. */
#define HAVE_ZLIB_H 1

/* Define to 1 if you have the '__asan_address_is_poisoned' function. */
/* #undef HAVE___ASAN_ADDRESS_IS_POISONED */

/* Define to 1 if you have the '__asan_region_is_poisoned' function. */
/* #undef HAVE___ASAN_REGION_IS_POISONED */

/* Define to 1 if your compiler supports __FUNCTION__ magic-macro. */
/* #undef HAVE___FUNCTION__ */

/* Define to 1 if your compiler supports __func__ magic-macro. */
#define HAVE___FUNC__ 1

/* Define to 1 if your compiler supports __PRETTY_FUNCTION__ magic-macro. */
/* #undef HAVE___PRETTY_FUNCTION__ */

/* Define to 1 if libmicrohttpd is compiled with HTTPS support. */
/* #undef HTTPS_SUPPORT */

/* Defined if libcurl supports AsynchDNS */
/* #undef LIBCURL_FEATURE_ASYNCHDNS */

/* Defined if libcurl supports IDN */
/* #undef LIBCURL_FEATURE_IDN */

/* Defined if libcurl supports IPv6 */
/* #undef LIBCURL_FEATURE_IPV6 */

/* Defined if libcurl supports KRB4 */
/* #undef LIBCURL_FEATURE_KRB4 */

/* Defined if libcurl supports libz */
/* #undef LIBCURL_FEATURE_LIBZ */

/* Defined if libcurl supports NTLM */
/* #undef LIBCURL_FEATURE_NTLM */

/* Defined if libcurl supports SSL */
/* #undef LIBCURL_FEATURE_SSL */

/* Defined if libcurl supports SSPI */
/* #undef LIBCURL_FEATURE_SSPI */

/* Defined if libcurl supports DICT */
/* #undef LIBCURL_PROTOCOL_DICT */

/* Defined if libcurl supports FILE */
/* #undef LIBCURL_PROTOCOL_FILE */

/* Defined if libcurl supports FTP */
/* #undef LIBCURL_PROTOCOL_FTP */

/* Defined if libcurl supports FTPS */
/* #undef LIBCURL_PROTOCOL_FTPS */

/* Defined if libcurl supports HTTP */
/* #undef LIBCURL_PROTOCOL_HTTP */

/* Defined if libcurl supports HTTPS */
/* #undef LIBCURL_PROTOCOL_HTTPS */

/* Defined if libcurl supports IMAP */
/* #undef LIBCURL_PROTOCOL_IMAP */

/* Defined if libcurl supports LDAP */
/* #undef LIBCURL_PROTOCOL_LDAP */

/* Defined if libcurl supports POP3 */
/* #undef LIBCURL_PROTOCOL_POP3 */

/* Defined if libcurl supports RTSP */
/* #undef LIBCURL_PROTOCOL_RTSP */

/* Defined if libcurl supports SMTP */
/* #undef LIBCURL_PROTOCOL_SMTP */

/* Defined if libcurl supports TELNET */
/* #undef LIBCURL_PROTOCOL_TELNET */

/* Defined if libcurl supports TFTP */
/* #undef LIBCURL_PROTOCOL_TFTP */

/* This is a Linux kernel */
#define LINUX 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to '1' if you have address sanitizer enabled */
/* #undef MHD_ASAN_ACTIVE */

/* Define to '1' if user memory poison is used */
/* #undef MHD_ASAN_POISON_ACTIVE */

/* The default HTTP Digest Auth default maximum nc (nonce count) value */
/* #undef MHD_DAUTH_DEF_MAX_NC_ */

/* The default HTTP Digest Auth default nonce timeout value (in seconds) */
/* #undef MHD_DAUTH_DEF_TIMEOUT_ */

/* Define to '1' to use fast (and larger) code version */
#define MHD_FAVOR_FAST_CODE 1

/* Define to '1' to use compact code version */
/* #undef MHD_FAVOR_SMALL_CODE */

/* Define to '1' if CPU_COUNT_S() function expects max CPU number as 'size'
   parameter */
/* #undef MHD_FUNC_CPU_COUNT_S_GETS_CPUS */

/* Define to 1 if you have suitable libmagic. */
/* #undef MHD_HAVE_LIBMAGIC */

/* Define to 1 if you have __builtin_bswap32() builtin function */
#define MHD_HAVE___BUILTIN_BSWAP32 1

/* Define to 1 if you have __builtin_bswap64() builtin function */
#define MHD_HAVE___BUILTIN_BSWAP64 1

/* Define to `1' if HTTPS require initialisation of libgcrypt */
/* #undef MHD_HTTPS_REQUIRE_GCRYPT */

/* Define to 1 if libmicrohttpd is compiled with MD5 hashing support. */
/* #undef MHD_MD5_SUPPORT */

/* Define to 1 if libmicrohttpd is compiled with MD5 hashing by TLS library.
   */
/* #undef MHD_MD5_TLSLIB */

/* Define to 1 to disable setting name on generated threads */
/* #undef MHD_NO_THREAD_NAMES */

/* tls plugins */
#define MHD_PLUGIN_INSTALL_PREFIX "/usr/local/lib/libmicrohttpd"

/* Define to 1 if libmicrohttpd is compiled with SHA-256 hashing support. */
/* #undef MHD_SHA256_SUPPORT */

/* Define to 1 if libmicrohttpd is compiled with SHA-256 hashing by TLS
   library. */
/* #undef MHD_SHA256_TLSLIB */

/* Define to 1 if libmicrohttpd is compiled with SHA-512/256 hashing support.
   */
/* #undef MHD_SHA512_256_SUPPORT */

/* Define to system default value of FD_SETSIZE macro */
#define MHD_SYS_FD_SETSIZE_ 1024

/* Define if you have usable `getsockname' function. */
#define MHD_USE_GETSOCKNAME 1

/* Define if you have usable PAGESIZE macro */
/* #undef MHD_USE_PAGESIZE_MACRO */

/* Define if you have PAGESIZE macro usable for static init */
/* #undef MHD_USE_PAGESIZE_MACRO_STATIC */

/* Define if you have usable PAGE_SIZE macro */
/* #undef MHD_USE_PAGE_SIZE_MACRO */

/* Define if you have PAGE_SIZE macro usable for static init */
/* #undef MHD_USE_PAGE_SIZE_MACRO_STATIC */

/* define to use pthreads */
#define MHD_USE_POSIX_THREADS 1

/* Define to 1 if you have properly working tsearch(), tfind() and tdelete()
   functions. */
#define MHD_USE_SYS_TSEARCH 1

/* define to use W32 threads */
/* #undef MHD_USE_W32_THREADS */

/* This is a MinGW system */
/* #undef MINGW */

/* This is a NetBSD system */
/* #undef NETBSD */

/* This is an OpenBSD system */
/* #undef OPENBSD */

/* This is a OS/390 system */
/* #undef OS390 */

/* This is an OS X system */
/* #undef OSX */

/* Some strange OS */
/* #undef OTHEROS */

/* Name of package */
#define PACKAGE "libmicrohttpd"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "libmicrohttpd@gnu.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "GNU libmicrohttpd"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "GNU libmicrohttpd 1.0.10"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libmicrohttpd"

/* Define to the home page for this package. */
#define PACKAGE_URL "https://www.gnu.org/software/libmicrohttpd/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.0.10"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of 'int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of 'int64_t', as computed by sizeof. */
#define SIZEOF_INT64_T 8

/* The size of 'size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 8

/* The size of `tv_sec' member of `struct timeval', as computed by sizeof */
#define SIZEOF_STRUCT_TIMEVAL_TV_SEC 8

/* The size of 'uint64_t', as computed by sizeof. */
#define SIZEOF_UINT64_T 8

/* The size of 'unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of 'unsigned long long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG_LONG 8

/* This is a Solaris system */
/* #undef SOLARIS */

/* This is a BSD system */
/* #undef SOMEBSD */

/* Define to 1 if all of the C89 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Define to 1 if libmicrohttpd is compiled with HTTP Upgrade support. */
/* #undef UPGRADE_SUPPORT */

/* Define to 1 if your kernel supports IPv6 and IPv6 is enabled and useful for
   testing. */
#define USE_IPV6_TESTING 1

/* Version number of package */
#define VERSION "1.0.10"

/* This is a Windows system */
/* #undef WINDOWS */

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define to 1 if necessary to make fseeko visible. */
/* #undef _LARGEFILE_SOURCE */

/* Define to 1 on platforms where this makes off_t a 64-bit type. */
/* #undef _LARGE_FILES */

/* defines how to decorate public symbols while building the library */
#define _MHD_EXTERN __attribute__((visibility("default"))) extern

/* Define to 1 to enable "heavy" test paths. */
/* #undef _MHD_HEAVY_TESTS */

/* Define to use eventFD for inter-thread communication */
#define _MHD_ITC_EVENTFD 1

/* Define to use pipe for inter-thread communication */
/* #undef _MHD_ITC_PIPE */

/* Define to use socketpair for inter-thread communication */
/* #undef _MHD_ITC_SOCKETPAIR */

/* Define to supported 'noreturn' function declaration */
#define _MHD_NORETURN [[noreturn]]

/* Define to 1 to enable "very heavy" test paths. */
/* #undef _MHD_VHEAVY_TESTS */

/* Define to prefix which will be used with MHD static inline functions. */
#define _MHD_static_inline static inline __attribute__((always_inline))

/* Need with solaris or errno does not work */
/* #undef _REENTRANT */

/* Number of bits in time_t, on hosts where this is settable. */
/* #undef _TIME_BITS */

/* Define to 1 on platforms where this makes time_t a 64-bit type. */
/* #undef __MINGW_USE_VC2005_COMPAT */

/* Define to 1 if C does not support variable-length arrays, and if the
   compiler does not already define this. */
/* #undef __STDC_NO_VLA__ */

/* Define to the name of the type which is used as a replacemnt for boolean
   type. */
/* #undef bool */

/* Define curl_free() as free() if our version of curl lacks curl_free. */
/* #undef curl_free */

/* Define to value interpreted by compiler as boolean "false", if "false" is
   not a keyword and not defined by headers. */
/* #undef false */

/* Define to '__inline__' or '__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to value interpreted by compiler as boolean "true", if "true" is not
   a keyword and not defined by headers. */
/* #undef true */
