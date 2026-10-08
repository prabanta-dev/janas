/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * catalog.c - the models Janas knows (see catalog.h). Every SHA-256 here
 * is one of MODELS.md; a model added here is added there too, and to
 * janas-try.sh's levels through janas-get. A "fast" profile holds the
 * levers measured on that model, every combination of them, to make it
 * faster by 2% or more each (8 Oct 2026, README "The fast profiles, as
 * measured"; the models larger than the cache on the same text, since each
 * lever changes the text a greedy reply takes and with it the experts read from
 * disk). Not
 * --head 4 on Qwen3.5-0.8B, which then answered "3" to "2+3"; nothing on
 * Gemma 4, whose keys are sixteen-bit, heads not six-bit and experts not
 * in planes.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "common/catalog.h"

const struct janas_model_entry janas_catalog[] = {
    {.name = "qwen3-0.6b",
     .what = "Qwen3-0.6B, dense - also the draft of Qwen3-4B",
     .repo = "unsloth/Qwen3-0.6B-GGUF",
     .parts = {"Qwen3-0.6B-Q4_K_M.gguf"},
     .part_sha =
         {"ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"},
     .gguf_gb = 0.40,
     .out = "qwen3-0.6b-q4km.jns",
     .flat_sha =
         "9174da760919b16b9f3a4ae8738c2d53605412f40bb2c0fe68d60174bf8a4fb5",
     .ram_gb = 2,
     .fast = "--head 4 --attention fast"},
    {.name = "qwen3-4b",
     .what = "Qwen3-4B, dense - the quick start",
     .repo = "Qwen/Qwen3-4B-GGUF",
     .parts = {"Qwen3-4B-Q4_K_M.gguf"},
     .part_sha =
         {"7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"},
     .gguf_gb = 2.5,
     .out = "qwen3-4b-q4km.jns",
     .flat_sha =
         "e8ed44bdd3c612ce3ec1204a4b9db671115efab4a2ada384c70cf6492d430291",
     .jns_sha =
         "1cbd8ebfdf14aee05777f5278e655382f040cdedc21d282e93806fc8cf279fc5",
     .ram_gb = 8,
     .draft = "qwen3-0.6b",
     .fast = "--head 4 --attention fast"},
    {.name = "qwen3.5-0.8b",
     .what = "Qwen3.5-0.8B, dense",
     .repo = "unsloth/Qwen3.5-0.8B-GGUF",
     .parts = {"Qwen3.5-0.8B-Q4_K_M.gguf"},
     .part_sha =
         {"bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"},
     .gguf_gb = 0.53,
     .out = "qwen3.5-0.8b-q4km.jns",
     .flat_sha =
         "75a3b62c4a10cad87a7ce64bf89485cfcc564fa949f2c5af8c7051615dc26124",
     .ram_gb = 2,
     .fast = "--attention fast"},
    {.name = "qwen3.5-2b",
     .what = "Qwen3.5-2B, dense, with its MTP block",
     .repo = "unsloth/Qwen3.5-2B-GGUF",
     .parts = {"Qwen3.5-2B-Q4_K_M.gguf"},
     .part_sha =
         {"aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"},
     .gguf_gb = 1.28,
     .out = "qwen3.5-2b-q4km.jns",
     .flat_sha =
         "286118ea3ad1903890a593582e22bed82859bf7c38b0ff9ed043050e5bd572a6",
     .extra = JANAS_EXTRA_HF2JNS,
     .extra_src = "hf:Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc",
     .extra_out = "qwen3.5-2b-mtp-q4k.jns",
     .extra_sha =
         "19c2d7eba1800acb10a06d7b38709ec92776e852cb2f25032ddd8977d17f2b05",
     .ram_gb = 4,
     .fast = "--head 4"},
    {.name = "qwen3.5-9b",
     .what = "Qwen3.5-9B, dense, with its MTP block",
     .repo = "unsloth/Qwen3.5-9B-GGUF",
     .parts = {"Qwen3.5-9B-Q4_K_M.gguf"},
     .part_sha =
         {"03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"},
     .gguf_gb = 5.68,
     .out = "qwen3.5-9b-q4km.jns",
     .flat_sha =
         "e40ec6c58ef2390c7a4ce8c9b7b6b384ed4e55c9980f646d1c98e83e2c018115",
     .extra = JANAS_EXTRA_HF2JNS,
     .extra_src = "hf:Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
     .extra_out = "qwen3.5-9b-mtp-q4k.jns",
     .extra_sha =
         "67f31f57dbc78ca8c02ef5e3aa137c5df3836a3decda20b6ea057d004089c361",
     .ram_gb = 8,
     .fast = "--head 4"},
    {.name = "qwen3-30b-a3b",
     .what = "Qwen3-30B-A3B, mixture of experts",
     .repo = "Qwen/Qwen3-30B-A3B-GGUF",
     .parts = {"Qwen3-30B-A3B-Q4_K_M.gguf"},
     .part_sha =
         {"0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5"},
     .gguf_gb = 18.6,
     .out = "qwen3-30b-a3b-q4km.jns",
     .flat_sha =
         "6906de51f2923b3be3a83eb040eaaeb090cba3389e9217648abe08bd678376e9",
     .jns_sha =
         "3fa0185b7eecd43e222d420f68aae28a86b3b90392fd7e4052b5cd337318e895",
     .ram_gb = 16,
     .fast = "--head 4 --attention fast --bits 4"},
    {.name = "qwen3.6-35b-a3b",
     .what = "Qwen3.6-35B-A3B, mixture of experts, MTP inside",
     .repo = "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
     .parts = {"Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"},
     .part_sha =
         {"b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"},
     .gguf_gb = 22.3,
     .out = "qwen3.6-35b-a3b-q4km.jns",
     .flat_sha =
         "31772037300f332d712f4ef8dd5e7251c7cd76cf6b9dab33b089394b7c079fb2",
     .jns_sha =
         "8feaeb66d93593b9876b564ca3878f2a9252e46b8d4e2d06caa4e19d43dc6464",
     .ram_gb = 16,
     .fast = "--head 4 --bits 4"},
    {.name = "qwen3-next-80b-a3b",
     .what = "Qwen3-Next-80B-A3B-Instruct, with its MTP block",
     .repo = "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF",
     .parts = {"Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"},
     .part_sha =
         {"d103b2733ec1012a52d01edda66b7e5c24ae50508c9f99f5297ea459ef3c061a"},
     .gguf_gb = 48.4,
     .out = "qwen3-next-80b-a3b-q4km.jns",
     .flat_sha =
         "fa9a50dd910de4e78064ecff17aef43286694277eea3a89a57e9921ca2bfd2da",
     .jns_sha =
         "fa3255c73106391e9cb5a97efd30f77a928fb4518d40060f795a8cff6656daed",
     .extra = JANAS_EXTRA_HF2JNS,
     .extra_src = "hf:Qwen/Qwen3-Next-80B-A3B-Instruct",
     .extra_out = "qwen3-next-80b-a3b-mtp-q4k.jns",
     .extra_sha =
         "5c9bc4cb6f739193fac5e4daa30c922f73255acf41443ecfc0123ef084584559",
     .ram_gb = 32,
     .fast = "--head 4 --bits 4"},
    {.name = "qwen3-coder-next",
     .what = "Qwen3-Coder-Next, mixture of experts, for code",
     .repo = "Qwen/Qwen3-Coder-Next-GGUF",
     .parts =
         {"Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
          "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00002-of-00004.gguf",
          "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00003-of-00004.gguf",
          "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf"},
     .part_sha =
         {"6bcfc9f9c37901eeb92172e2ab871224dab36a453d263bcb2547f737409534da",
          "817def0691ee9d08bf3dc4444be7aed29c9e52091e8fa9d97901ce7e7f6f01d3",
          "23aa634d47dca9b4ca3ea249384e6f01951b24c83cdc076f37f6f43d6c99883f",
          "249c768cc5f130dc731567d6edcbdacc48e14dec9e02c5dbe2b2185d2c5bdb2b"},
     .gguf_gb = 48.4,
     .out = "qwen3-coder-next-q4km.jns",
     .flat_sha =
         "5912d964247002473eff1811058b4ffadf7b9bb47ae336c486cdf2964de68382",
     .jns_sha =
         "0276b0697a05075fec8ab076a757205537804ff41dd7e9dadace7c6993763563",
     .ram_gb = 32,
     .fast = "--head 4 --bits 4"},
    {.name = "qwen3-embedding-0.6b",
     .what = "Qwen3-Embedding-0.6B, for janas-server --embedding-model",
     .repo = "Qwen/Qwen3-Embedding-0.6B-GGUF",
     .parts = {"Qwen3-Embedding-0.6B-Q8_0.gguf"},
     .part_sha =
         {"06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"},
     .gguf_gb = 0.64,
     .out = "qwen3-embedding-0.6b-q8.jns",
     .flat_sha =
         "1e8eb8cbeba7a1a657354d259c5730bb79f6087160a9ca8da901bd08bdbe943f",
     .ram_gb = 2,
     .not_chat = 1},
    {.name = "gemma-4-e2b",
     .what = "Gemma-4-E2B-it, with its assistant",
     .repo = "unsloth/gemma-4-E2B-it-GGUF",
     .parts = {"gemma-4-E2B-it-Q4_K_M.gguf"},
     .part_sha =
         {"740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"},
     .gguf_gb = 3.1,
     .out = "gemma-4-e2b-it-q4km.jns",
     .flat_sha =
         "1f18e08878df41f377d563b38048d41841c7973ddea845cd9672fcf23cbd864b",
     .extra = JANAS_EXTRA_GGUF,
     .extra_src = "mtp-gemma-4-E2B-it.gguf",
     .extra_src_sha =
         "9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919",
     .extra_out = "gemma-4-e2b-it-mtp.jns",
     .extra_sha =
         "79df1f3c817542b3867a0940b1a5b7330d478bddb4102ec90ff817334877cd62",
     .ram_gb = 4},
    {.name = "gemma-4-e4b",
     .what = "Gemma-4-E4B-it, with its assistant",
     .repo = "unsloth/gemma-4-E4B-it-GGUF",
     .parts = {"gemma-4-E4B-it-Q4_K_M.gguf"},
     .part_sha =
         {"85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87"},
     .gguf_gb = 5.0,
     .out = "gemma-4-e4b-it-q4km.jns",
     .flat_sha =
         "7ec828e9bdc1ada2eda907e93b87072f3180f2e8cba5ccedcd360d1a53858675",
     .extra = JANAS_EXTRA_GGUF,
     .extra_src = "mtp-gemma-4-E4B-it.gguf",
     .extra_src_sha =
         "b6a723115efa510d3b3215db1e26790dae84cd08c2134a764f3d194f1f0c3376",
     .extra_out = "gemma-4-e4b-it-mtp.jns",
     .extra_sha =
         "7c6fdb74ba1eb11f9d47b15097a09cce0142aea380a41f1986ba7b8ec4785b53",
     .ram_gb = 8},
    {.name = "gemma-4-12b",
     .what = "Gemma-4-12B-it, dense, with its assistant",
     .repo = "unsloth/gemma-4-12b-it-GGUF",
     .parts = {"gemma-4-12b-it-Q4_K_M.gguf"},
     .part_sha =
         {"0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42"},
     .gguf_gb = 7.1,
     .out = "gemma-4-12b-it-q4km.jns",
     .flat_sha =
         "cae09e04598891dea6e9cd93fb496fbde08d822ac35de4ff45dbfab1ea25c4dd",
     .extra = JANAS_EXTRA_GGUF,
     .extra_src = "mtp-gemma-4-12b-it.gguf",
     .extra_src_sha =
         "145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b",
     .extra_out = "gemma-4-12b-it-mtp.jns",
     .extra_sha =
         "35e640f20dd74e862df9c1d949ad4edca9f58beef0cb29e99cccc6c99297826f",
     .ram_gb = 16},
    {.name = "gemma-4-26b-a4b",
     .what = "Gemma-4-26B-A4B-it, mixture of experts, with its assistant",
     .repo = "unsloth/gemma-4-26B-A4B-it-GGUF",
     .parts = {"gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
     .part_sha =
         {"f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f"},
     .gguf_gb = 16.9,
     .out = "gemma-4-26b-a4b-it-udq4km.jns",
     .flat_sha =
         "35c3845bbc21de52bf630d3c7455b4bb5379c34487b5925e020ec2e573931666",
     .extra = JANAS_EXTRA_GGUF,
     .extra_src = "mtp-gemma-4-26B-A4B-it.gguf",
     .extra_src_sha =
         "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b",
     .extra_out = "gemma-4-26b-it-mtp.jns",
     .extra_sha =
         "6ba24ee91fe6fe904b29919acbb654a66d9caa0aba67db1df9f921a76e5bcedf",
     .ram_gb = 16},
};

const int janas_catalog_n =
    (int)(sizeof(janas_catalog) / sizeof(janas_catalog[0]));

const struct janas_model_entry *janas_catalog_find(const char *name)
{
    for (int i = 0; i < janas_catalog_n; i++)
        if (strcmp(janas_catalog[i].name, name) == 0)
            return &janas_catalog[i];
    return NULL;
}

int janas_models_dir(char *buf, size_t len)
{
    const char *e = getenv("JANAS_MODELS"), *x = getenv("XDG_DATA_HOME"),
               *h = getenv("HOME");
    int n;
    if (e && *e)
        n = snprintf(buf, len, "%s", e);
    else if (x && *x)
        n = snprintf(buf, len, "%s/janas/models", x);
    else if (h && *h)
        n = snprintf(buf, len, "%s/.local/share/janas/models", h);
    else
        return -1;
    return n > 0 && (size_t)n < len ? 0 : -1;
}

int janas_mkdirs(const char *dir)
{
    char p[4096];
    int n = snprintf(p, sizeof(p), "%s", dir);
    if (n <= 0 || (size_t)n >= sizeof(p))
        return -1;
    for (char *c = p + 1; *c; c++)
        if (*c == '/') {
            *c = 0;
            if (mkdir(p, 0755) != 0 && errno != EEXIST)
                return -1;
            *c = '/';
        }
    if (mkdir(p, 0755) != 0 && errno != EEXIST)
        return -1;
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : -1;
}
