/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-system's answers (see system.h and
 * services/common/template.h): the text the user reads, filled with the
 * data in English here and in the user's language by a client that
 * translates the layout once (janas-chat), and the brief, all the model
 * reads of it. Names of programs, paths, messages of the logs and of the
 * packages stay as they are.
 */
#include "system.h"

#define SIZE_USED "{{used.v}} {{used.u}}"
#define SIZE_TOTAL "{{total.v}} {{total.u}}"
#define SIZE_AVAIL "{{avail.v}} {{avail.u}}"
#define WHEN                                                                   \
    "{{#when.today}}today{{/when.today}}{{^when.today}}{{when.day}}/"          \
    "{{when.mon}}{{/when.today}} {{when.at}}"
#define SHOWN                                                                  \
    " Shown to the user: do not repeat it; answer in a line or two, in the "   \
    "user's language."

const char SY_STATUS_LAYOUT[] =
    "{{host}}: {{os}}{{#kernel}}, {{kernel}}{{/kernel}}; on for "
    "{{#up.d}}{{up.d}} d {{/up.d}}{{up.h}} h {{up.m}} min.\n"
    "Processor: {{cpu}}, {{cpus}} threads{{#has_cpu}}, {{cpu_pct}}% in "
    "use{{#sys_pct}} ({{sys_pct}}% by the kernel){{/sys_pct}}{{/has_cpu}}{{#has_load}}; load {{load1}}, {{load5}}, {{load15}} "
    "(1, 5, 15 min){{/has_load}}.\n"
    "Memory: {{mem.used.v}} {{mem.used.u}} used of {{mem.total.v}} "
    "{{mem.total.u}} ({{mem.pct}}%), {{mem.avail.v}} {{mem.avail.u}} "
    "available{{#swap.has}}; swap: {{swap.used.v}} {{swap.used.u}} of "
    "{{swap.total.v}} {{swap.total.u}}{{/swap.has}}.\n"
    "{{#psi.high}}\n"
    "Waiting, of the last 10 seconds: for the processor {{psi.cpu}}%, for "
    "memory {{psi.mem}}%, for the disks {{psi.io}}%.\n"
    "{{/psi.high}}\n"
    "Disks:\n"
    "{{#disks}}\n"
    "  {{mount}}: " SIZE_USED " of " SIZE_TOTAL " ({{pct}}%), " SIZE_AVAIL
    " free{{#full}} - nearly full{{/full}}\n"
    "{{/disks}}\n"
    "{{#bat.present}}\n"
    "Battery: {{bat.pct}}%, {{bat.state|bat}}{{#bat.has_time}}, "
    "{{#bat.draining}}{{bat.time.h}} h {{bat.time.m}} min "
    "left{{/bat.draining}}{{^bat.draining}}full in {{bat.time.h}} h "
    "{{bat.time.m}} min{{/bat.draining}}{{/bat.has_time}}{{#bat.health}}; "
    "capacity: {{bat.health}}% of what it was new{{/bat.health}}.\n"
    "{{/bat.present}}\n"
    "{{#n_temps}}\n"
    "Temperatures:\n"
    "{{/n_temps}}\n"
    "{{#temps}}\n"
    "  {{kind|sensor}}{{#label}} {{label}}{{/label}}: {{c}} °C{{#hot}}, near "
    "its limit of {{high}} °C{{/hot}}\n"
    "{{/temps}}\n"
    "---\n"
    "bat.charging = charging\n"
    "bat.discharging = on battery\n"
    "bat.full = full\n"
    "bat.idle = plugged in, not charging\n"
    "bat.unknown = state not told\n"
    "sensor.cpu = processor\n"
    "sensor.core = hottest core\n"
    "sensor.disk = disk\n"
    "sensor.memory = memory\n"
    "sensor.wifi = Wi-Fi card\n"
    "sensor.board = motherboard\n"
    "sensor.gpu = graphics card\n"
    "sensor.battery = battery\n"
    "sensor.chipset = chipset\n"
    "sensor.other = other sensor\n";

const char SY_STATUS_BRIEF[] =
    "{{host}} ({{os}}), on {{up.d}} d {{up.h}} h. CPU {{cpu_pct}}% (kernel "
    "{{sys_pct}}%) of "
    "{{cpus}} threads{{#has_load}}, load {{load1}}/{{load5}}/"
    "{{load15}}{{/has_load}}. Memory {{mem.pct}}% used, {{mem.avail.v}} "
    "{{mem.avail.u}} free{{#swap.has}}, swap {{swap.pct}}%{{/swap.has}}."
    "{{#psi.high}} Waiting: CPU {{psi.cpu}}%, memory {{psi.mem}}%, disks "
    "{{psi.io}}%.{{/psi.high}} Disks: {{#disks}}{{mount}} {{pct}}% "
    "({{avail.v}} {{avail.u}} free{{#full}}, NEARLY FULL{{/full}}); "
    "{{/disks}}{{#bat.present}}battery {{bat.pct}}% {{bat.state}}{{#bat.worn}}"
    ", worn ({{bat.health}}%){{/bat.worn}}; {{/bat.present}}{{#temps}}"
    "{{kind}} {{c}} C{{#hot}} NEAR ITS LIMIT{{/hot}}; {{/temps}}Shown to the "
    "user: do not repeat it; answer in a line or two, in the user's "
    "language, telling only what is wrong (a disk nearly full, memory "
    "short, a temperature near its limit, a worn battery), or that all is "
    "well.";

const char SY_DISKS_LAYOUT[] =
    "Disks{{#more}} (the first {{n}}){{/more}}:\n"
    "{{#disks}}\n"
    "  {{mount}} ({{dev}}, {{fs}}{{#ro}}, read only{{/ro}}): " SIZE_USED
    " used of " SIZE_TOTAL " ({{pct}}%), " SIZE_AVAIL
    " free{{#full}} - nearly full{{/full}}\n"
    "{{/disks}}\n";

const char SY_DISKS_BRIEF[] =
    "Disks: {{#disks}}{{mount}} " SIZE_USED " of " SIZE_TOTAL
    " ({{pct}}%{{#full}}, NEARLY FULL{{/full}}); {{/disks}}" SHOWN;

const char SY_USAGE_LAYOUT[] =
    "{{path}}: {{total.v}} {{total.u}} in {{files}} files{{#cut}}, at "
    "least: the time ran out after {{secs}} s{{/cut}}; the disk has "
    "{{fs_avail.v}} {{fs_avail.u}} free of {{fs_total.v}} "
    "{{fs_total.u}}.\n"
    "{{#items}}\n"
    "  {{size.v}} {{size.u}} ({{pct}}%)  {{entry}}{{#dir}}/, {{files}} "
    "files{{/dir}}\n"
    "{{/items}}\n"
    "{{#more}}\n"
    "  ... and {{more}} more entries.\n"
    "{{/more}}\n"
    "{{#unread}}\n"
    "Directories that could not be read, left out: {{unread}}.\n"
    "{{/unread}}\n";

const char SY_USAGE_BRIEF[] =
    "{{path}}: {{total.v}} {{total.u}}{{#cut}} at least (time ran "
    "out){{/cut}}, disk free {{fs_avail.v}} {{fs_avail.u}}. Biggest: "
    "{{#items}}{{entry}}{{#dir}}/{{/dir}} {{size.v}} {{size.u}}; "
    "{{/items}}" SHOWN;

const char SY_PROCS_LAYOUT[] =
    "{{#name}}Processes with \"{{name}}\" in their name: {{matched}} of "
    "{{total}}{{/name}}{{^name}}The processes using the most "
    "{{by|by}}, of {{total}}{{/name}}:\n"
    "{{#items}}\n"
    "  {{pname}} (pid {{pid}}, {{user}}): processor {{cpu}}%, memory "
    "{{mem.v}} {{mem.u}} ({{mem_pct}}%), running for "
    "{{#age.d}}{{age.d}} d {{/age.d}}{{age.h}} h {{age.m}} min\n"
    "{{/items}}\n"
    "{{^n}}\n"
    "  none.\n"
    "{{/n}}\n"
    "{{#more}}\n"
    "  ... and {{more}} more.\n"
    "{{/more}}\n"
    "---\n"
    "by.cpu = processor\n"
    "by.memory = memory\n";

const char SY_PROCS_BRIEF[] =
    "{{#name}}Processes named like {{name}}: {{matched}}.{{/name}}"
    "{{^name}}Top by {{by}} of {{total}}:{{/name}} {{#items}}{{pname}} "
    "(pid {{pid}}) CPU {{cpu}}% mem {{mem.v}} {{mem.u}}; {{/items}}"
    "CPU % is of the whole processor." SHOWN;

const char SY_ERRORS_LAYOUT[] =
    "{{#boot}}Errors since the computer started{{/boot}}{{^boot}}Errors in "
    "the last {{hours}} hours{{/boot}}: {{#cut}}at least {{/cut}}{{total}}"
    "{{#more}}, the latest "
    "{{n}} kinds shown{{/more}}.\n"
    "{{#user_only}}\n"
    "Only those of your own programs and session: the system's need the "
    "group adm or systemd-journal.\n"
    "{{/user_only}}\n"
    "{{#items}}\n"
    "  " WHEN " {{who}}{{#again}} ({{count}} times){{/again}}: {{text}}\n"
    "{{/items}}\n"
    "{{#n_failed}}\n"
    "Services that failed:\n"
    "{{/n_failed}}\n"
    "{{#failed}}\n"
    "  {{unit}}{{#again}} ({{count}} times){{/again}}\n"
    "{{/failed}}\n"
    "{{^n_failed}}\n"
    "No service has failed.\n"
    "{{/n_failed}}\n";

const char SY_ERRORS_BRIEF[] =
    "{{#boot}}Since boot{{/boot}}{{^boot}}Last {{hours}} h{{/boot}}: "
    "{{total}} errors{{#user_only}} (user's only){{/user_only}}. "
    "{{#items}}{{who}} x{{count}}: {{text}}; {{/items}}Failed services: "
    "{{n_failed}}{{#failed}} {{unit}};{{/failed}} Shown to the user: do not "
    "repeat the list; answer in a line or two, in the user's language, "
    "saying which matter, if any.";

const char SY_UPDATES_LAYOUT[] =
    "{{#total}}Updates available: {{total}}{{#security}}, {{security}} of "
    "them for security{{/security}}{{#more}} (the first {{n}} "
    "listed){{/more}}{{/total}}{{^total}}No update available{{/total}}, "
    "by {{tool}}'s list{{#checked.has}} as of "
    "{{checked.day}}/"
    "{{checked.mon}} {{checked.at}}{{#checked.ago_d}}, {{checked.ago_d}} "
    "days ago{{/checked.ago_d}}{{/checked.has}}.\n"
    "{{#items}}\n"
    "  {{pkg}} {{#from}}{{from}} -> {{/from}}{{to}}{{#sec}} "
    "(security){{/sec}}\n"
    "{{/items}}\n"
    "{{#refresh}}\n"
    "The list is the one last fetched; it is fetched again with: "
    "{{refresh}}\n"
    "{{/refresh}}\n";

const char SY_UPDATES_BRIEF[] =
    "{{tool}}: {{total}} updates, {{security}} for security; list as of "
    "{{checked.day}}/{{checked.mon}}, {{checked.ago_d}} days ago. "
    "{{#items}}{{pkg}}{{#sec}} (security){{/sec}}; {{/items}}" SHOWN;

const char SY_NET_LAYOUT[] =
    "{{#items}}\n"
    "{{name}} ({{kind|iface}}, {{state|link}}){{#gw}}, the way out to the "
    "internet{{/gw}}:{{#addrs}} {{.}}{{/addrs}}{{^addrs}} no "
    "address{{/addrs}}\n"
    "{{#speed}}\n"
    "  speed: {{speed}} Mb/s\n"
    "{{/speed}}\n"
    "{{#signal}}\n"
    "  signal: {{signal}} dBm ({{wifi_pct}}%)\n"
    "{{/signal}}\n"
    "  received {{rx.v}} {{rx.u}}, sent {{tx.v}} {{tx.u}}\n"
    "{{/items}}\n"
    "{{#gateway}}Gateway: {{gateway}}{{/gateway}}{{^gateway}}No gateway: "
    "no way out to the internet{{/gateway}}{{#n_dns}}; name "
    "servers:{{/n_dns}}{{#dns}} {{.}}{{/dns}}.\n"
    "---\n"
    "iface.wifi = Wi-Fi\n"
    "iface.ethernet = cable\n"
    "iface.virtual = virtual\n"
    "link.up = connected\n"
    "link.down = not connected\n"
    "link.dormant = waiting\n"
    "link.lowerlayerdown = not connected\n"
    "link.unknown = state not told\n";

const char SY_NET_BRIEF[] =
    "{{#items}}{{name}} {{kind}} {{state}}{{#gw}} (default "
    "route){{/gw}}{{#addrs}} {{.}}{{/addrs}}{{#signal}}, signal "
    "{{signal}} dBm{{/signal}}; {{/items}}gateway {{gateway}}." SHOWN;
