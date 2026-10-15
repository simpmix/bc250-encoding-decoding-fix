#!/bin/bash
# Look for credentials in the files about to leave the machine.
#
# WHY IT EXISTS
# On 16/08/2026 a root password ended up in a public MTSistemi repository and
# stayed there four days. Nobody saw it because nothing was looking: CodeQL
# looks for defects in code, and GitHub's secret scanning recognises the known
# FORMATS of tokens (ghp_, AWS keys). An ordinary password it does not.
#
# ⚠️ THIS FILE CONTAINS NO SECRET, and must never contain one. It looks for the
# SHAPE of a credential, never for a value: a detector that searched for the
# real password would itself be the leak.
#
#   usage:  scripts/check-secrets.sh              the whole tracked tree
#           scripts/check-secrets.sh --outgoing   only what is not on the upstream yet
set -u
cd "$(dirname "$0")/.."

# The shapes. No values, only structures.
#   password="…"            a password written by hand (with quotes:
#                           password=VARIABLE has none and is fine). The dots
#                           are ONE character on purpose: the shape wants at
#                           least six between the quotes, so this line does not
#                           flag itself.
#   sshpass, -p, a value    the classic way to hand ssh a password, when the
#                           value is written there; a "$VAR" after -p reads it
#                           from the environment and is fine
#   BEGIN ... PRIVATE KEY   a private key
#   ghp_ / github_pat_      GitHub tokens
#   AKIA...                 AWS keys
# The value must be a LITERAL: a quote right after the equals sign, then
# characters that are not $ (a variable), / (a path) or < (a placeholder).
SHAPES='(password|passwd|passphrase)[[:space:]]*=[[:space:]]*["'"'"'][^"'"'"'$/<][^"'"'"']{5,}["'"'"']|sshpass[[:space:]]+-p[[:space:]]*["'"'"']?[^"'"'"'$[:space:]]|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}'

# What looks like a secret and is not: placeholders in help texts, the lines
# that SEARCH for passwords (this file), /etc/passwd, and concatenated strings
# ("...&password=" + variable builds a URL; the value is not in the file).
HARMLESS='<password>|YOUR-PASSWORD|example|xxxxx|\*\*\*\*|grep|etc/passwd|SHAPES=|HARMLESS=|"[[:space:]]*\+|\+[[:space:]]*"'

EXCLUDED=(':!*.png' ':!*.jpg' ':!*.ico' ':!*.woff2' ':!*.bin' ':!*.rom' ':!*.uf2' ':!*.deb' ':!*.iso')

if [ "${1:-}" = "--outgoing" ]; then
    BASE=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || echo "")
    if [ -z "$BASE" ]; then
        echo "no upstream branch: checking the whole tree"
        FOUND=$(git grep -nEI "$SHAPES" -- . "${EXCLUDED[@]}" 2>/dev/null || true)
    else
        echo "checking only what is not on $BASE yet"
        FOUND=$(git diff -U0 "$BASE"..HEAD -- . "${EXCLUDED[@]}" 2>/dev/null \
                | grep '^+' | grep -vE '^\+\+\+' | grep -nEI "$SHAPES" || true)
    fi
else
    FOUND=$(git grep -nEI "$SHAPES" -- . "${EXCLUDED[@]}" 2>/dev/null || true)
fi

FOUND=$(printf '%s\n' "$FOUND" | grep -vEi "$HARMLESS" | grep -v '^$' || true)

if [ -n "$FOUND" ]; then
    echo
    echo "CREDENTIALS FOUND - do not push:"
    printf '%s\n' "$FOUND" | cut -c1-140 | sed 's/^/   /'
    echo
    echo "   Credentials belong in a file outside the repository, read at run time."
    exit 1
fi

echo "no credentials in the files"
exit 0
