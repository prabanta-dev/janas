/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tex.h - a formula a model writes between dollars, $25.0625^\circ C$, as
 * the text a terminal can show: 25.0625°C. A page draws it with KaTeX; a
 * terminal has only characters, and Unicode has most of what a reply's
 * formulas use - Greek letters, the operators, the digits raised and
 * lowered. What has no character of its own keeps a plain form: x^{n+1}
 * becomes x^(n+1), \frac{a+b}{2} becomes (a+b)/2.
 */
#ifndef JANAS_CHAT_TEX_H
#define JANAS_CHAT_TEX_H

#include <string.h>

struct tex_out {
    char *p;
    size_t n, cap;
};

static void tex_put(struct tex_out *o, const char *s)
{
    size_t k = strlen(s);
    if (o->n + k < o->cap) {
        memcpy(o->p + o->n, s, k);
        o->n += k;
        o->p[o->n] = 0;
    }
}

static void tex_putc(struct tex_out *o, char c)
{
    char s[2] = {c, 0};
    tex_put(o, s);
}

/* the commands with a character of their own */
static const char *tex_symbol(const char *name)
{
    static const char *const tab[][2] = {
        {"alpha", "α"},       {"beta", "β"},
        {"gamma", "γ"},       {"delta", "δ"},
        {"epsilon", "ε"},     {"varepsilon", "ε"},
        {"zeta", "ζ"},        {"eta", "η"},
        {"theta", "θ"},       {"iota", "ι"},
        {"kappa", "κ"},       {"lambda", "λ"},
        {"mu", "μ"},          {"nu", "ν"},
        {"xi", "ξ"},          {"pi", "π"},
        {"rho", "ρ"},         {"sigma", "σ"},
        {"tau", "τ"},         {"phi", "φ"},
        {"varphi", "φ"},      {"chi", "χ"},
        {"psi", "ψ"},         {"omega", "ω"},
        {"Gamma", "Γ"},       {"Delta", "Δ"},
        {"Theta", "Θ"},       {"Lambda", "Λ"},
        {"Pi", "Π"},          {"Sigma", "Σ"},
        {"Phi", "Φ"},         {"Psi", "Ψ"},
        {"Omega", "Ω"},       {"circ", "°"},
        {"degree", "°"},      {"times", "×"},
        {"cdot", "·"},        {"div", "÷"},
        {"pm", "±"},          {"mp", "∓"},
        {"le", "≤"},          {"leq", "≤"},
        {"ge", "≥"},          {"geq", "≥"},
        {"ne", "≠"},          {"neq", "≠"},
        {"approx", "≈"},      {"sim", "∼"},
        {"equiv", "≡"},       {"infty", "∞"},
        {"to", "→"},          {"rightarrow", "→"},
        {"leftarrow", "←"},   {"Rightarrow", "⇒"},
        {"Leftarrow", "⇐"},   {"leftrightarrow", "↔"},
        {"iff", "⇔"},         {"implies", "⇒"},
        {"sum", "∑"},         {"prod", "∏"},
        {"int", "∫"},         {"partial", "∂"},
        {"nabla", "∇"},       {"in", "∈"},
        {"notin", "∉"},       {"subset", "⊂"},
        {"subseteq", "⊆"},    {"cup", "∪"},
        {"cap", "∩"},         {"emptyset", "∅"},
        {"forall", "∀"},      {"exists", "∃"},
        {"neg", "¬"},         {"land", "∧"},
        {"lor", "∨"},         {"oplus", "⊕"},
        {"ldots", "…"},       {"dots", "…"},
        {"cdots", "⋯"},       {"sqrt", "√"},
        {"angle", "∠"},       {"perp", "⊥"},
        {"parallel", "∥"},    {"propto", "∝"},
        {"hbar", "ħ"},        {"ell", "ℓ"},
        {"quad", " "},        {"qquad", "  "},
        {",", " "},           {";", " "},
        {":", " "},           {"!", ""},
        {"%", "%"},           {"$", "$"},
        {"{", "{"},           {"}", "}"},
        {"_", "_"},           {"#", "#"},
        {"&", "&"},           {"\\", " "},
        {"left", ""},         {"right", ""},
        {"displaystyle", ""}, {"limits", ""},
    };
    for (size_t i = 0; i < sizeof tab / sizeof tab[0]; i++)
        if (!strcmp(name, tab[i][0]))
            return tab[i][1];
    return NULL;
}

/* a character raised (up) or lowered, or NULL when Unicode has none */
static const char *tex_script(char c, int up)
{
    static const char *const sup_d[] = {"⁰", "¹", "²", "³", "⁴",
                                        "⁵", "⁶", "⁷", "⁸", "⁹"};
    static const char *const sub_d[] = {"₀", "₁", "₂", "₃", "₄",
                                        "₅", "₆", "₇", "₈", "₉"};
    if (c >= '0' && c <= '9')
        return up ? sup_d[c - '0'] : sub_d[c - '0'];
    switch (c) {
    case '+':
        return up ? "⁺" : "₊";
    case '-':
        return up ? "⁻" : "₋";
    case '=':
        return up ? "⁼" : "₌";
    case '(':
        return up ? "⁽" : "₍";
    case ')':
        return up ? "⁾" : "₎";
    case 'n':
        return up ? "ⁿ" : "ₙ";
    case 'i':
        return up ? "ⁱ" : "ᵢ";
    case 'a':
        return up ? NULL : "ₐ";
    case 'e':
        return up ? NULL : "ₑ";
    case 'o':
        return up ? NULL : "ₒ";
    case 'x':
        return up ? NULL : "ₓ";
    case 'j':
        return up ? NULL : "ⱼ";
    case 'k':
        return up ? NULL : "ₖ";
    case 'm':
        return up ? NULL : "ₘ";
    case 't':
        return up ? NULL : "ₜ";
    default:
        return NULL;
    }
}

static void tex_conv(const char *s, size_t n, struct tex_out *o);

/* a group's extent: {...} with its nesting, or one command, or one char */
static size_t tex_arg(const char *s, size_t n, size_t *start, size_t *len)
{
    size_t i = 0;
    while (i < n && s[i] == ' ')
        i++;
    if (i >= n) {
        *start = i, *len = 0;
        return i;
    }
    if (s[i] == '{') {
        int depth = 1;
        size_t j = i + 1;
        while (j < n && depth) {
            if (s[j] == '\\' && j + 1 < n)
                j++;
            else if (s[j] == '{')
                depth++;
            else if (s[j] == '}')
                depth--;
            j++;
        }
        *start = i + 1;
        *len = (j - 1) - (i + 1);
        return j;
    }
    if (s[i] == '\\') {
        size_t j = i + 1;
        while (j < n &&
               ((s[j] >= 'a' && s[j] <= 'z') || (s[j] >= 'A' && s[j] <= 'Z')))
            j++;
        if (j == i + 1 && j < n)
            j++;
        *start = i, *len = j - i;
        return j;
    }
    /* one character, a UTF-8 sequence whole */
    size_t j = i + 1;
    while (j < n && ((unsigned char)s[j] & 0xc0) == 0x80)
        j++;
    *start = i, *len = j - i;
    return j;
}

/* x^{...} or x_{...}: raised or lowered where every character can be */
static void tex_scripted(const char *s, size_t n, struct tex_out *o, int up)
{
    if (up && n == 5 && !memcmp(s, "\\circ", 5)) {
        tex_put(o, "°");
        return;
    }
    int all = n > 0;
    for (size_t i = 0; i < n && all; i++)
        all = tex_script(s[i], up) != NULL;
    if (all) {
        for (size_t i = 0; i < n; i++)
            tex_put(o, tex_script(s[i], up));
        return;
    }
    tex_putc(o, up ? '^' : '_');
    int one = n == 1 || s[0] == '\\'; /* x^2 aside, x^\alpha */
    if (!one)
        tex_putc(o, '(');
    tex_conv(s, n, o);
    if (!one)
        tex_putc(o, ')');
}

/* a fraction's part: in brackets unless it is one thing */
static void tex_part(const char *s, size_t n, struct tex_out *o)
{
    int simple = 1;
    for (size_t i = 0; i < n && simple; i++)
        simple = !strchr("+-*/ =", s[i]);
    if (!simple)
        tex_putc(o, '(');
    tex_conv(s, n, o);
    if (!simple)
        tex_putc(o, ')');
}

static void tex_conv(const char *s, size_t n, struct tex_out *o)
{
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        if (c == '^' || c == '_') {
            size_t a, l;
            const char *r = s + i + 1;
            size_t e = tex_arg(r, n - i - 1, &a, &l);
            tex_scripted(r + a, l, o, c == '^');
            i += 1 + e;
            continue;
        }
        if (c == '{' || c == '}') {
            i++;
            continue;
        }
        if (c == '\\') {
            size_t j = i + 1;
            while (j < n && ((s[j] >= 'a' && s[j] <= 'z') ||
                             (s[j] >= 'A' && s[j] <= 'Z')))
                j++;
            if (j == i + 1 && j < n)
                j++; /* \, \{ \% and the like */
            char name[32];
            size_t k =
                j - i - 1 < sizeof name - 1 ? j - i - 1 : sizeof name - 1;
            memcpy(name, s + i + 1, k);
            name[k] = 0;
            i = j;
            if (!strcmp(name, "frac") || !strcmp(name, "dfrac") ||
                !strcmp(name, "tfrac")) {
                size_t a1, l1, a2, l2;
                size_t e1 = tex_arg(s + i, n - i, &a1, &l1);
                size_t e2 = tex_arg(s + i + e1, n - i - e1, &a2, &l2);
                tex_part(s + i + a1, l1, o);
                tex_putc(o, '/');
                tex_part(s + i + e1 + a2, l2, o);
                i += e1 + e2;
                continue;
            }
            if (!strcmp(name, "text") || !strcmp(name, "mathrm") ||
                !strcmp(name, "mathbf") || !strcmp(name, "mathit") ||
                !strcmp(name, "operatorname") || !strcmp(name, "textbf") ||
                !strcmp(name, "mathsf") || !strcmp(name, "boldsymbol")) {
                size_t a, l, e = tex_arg(s + i, n - i, &a, &l);
                if (!strcmp(name, "text"))
                    for (size_t q = 0; q < l; q++)
                        tex_putc(o, s[i + a + q]);
                else
                    tex_conv(s + i + a, l, o);
                i += e;
                continue;
            }
            const char *sym = tex_symbol(name);
            if (!strcmp(name, "sqrt")) {
                size_t a, l, e = tex_arg(s + i, n - i, &a, &l);
                tex_put(o, "√");
                tex_part(s + i + a, l, o);
                i += e;
                continue;
            }
            if (sym)
                tex_put(o, sym);
            else if (!strcmp(name, "sin") || !strcmp(name, "cos") ||
                     !strcmp(name, "tan") || !strcmp(name, "log") ||
                     !strcmp(name, "ln") || !strcmp(name, "exp") ||
                     !strcmp(name, "max") || !strcmp(name, "min") ||
                     !strcmp(name, "lim") || !strcmp(name, "det"))
                tex_put(o, name);
            else
                tex_put(o, name); /* unknown: its name, at least */
            continue;
        }
        /* a formula's spaces are TeX's to drop: after a degree sign
           (75^\circ C), at least, they would show */
        if (c == ' ' && o->n >= 2 && !memcmp(o->p + o->n - 2, "°", 2)) {
            i++;
            continue;
        }
        tex_putc(o, c);
        i++;
    }
}

#endif
