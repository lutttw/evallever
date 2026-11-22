# -*- coding: utf-8 -*-
"""EvalLever bundle integrity verifier.

Recomputes sha256 for every manifest entry and compares against MANIFEST.txt.
Entries listed in the 'Excluded from this reviewer/git copy' block are
deliberately absent and are skipped (not counted as failures).

Usage:  python verify_bundle.py          (run from the bundle root)
Exit:   0 = all verified, 1 = any mismatch/missing.
"""
import hashlib
import os
import re
import sys

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    man = "MANIFEST.txt"
    if not os.path.isfile(man):
        print("MANIFEST.txt not found; run from the bundle root.")
        return 1
    ok = bad = missing = excluded = 0
    problems = []
    with open(man, encoding="utf-8") as f:
        for line in f:
            if line.startswith("  "):
                # indented lines belong to the explicit excluded block
                m = re.match(r"^\s+([0-9a-f]{64})\s+[\d,]+\s+(\S.*)$", line)
                if m:
                    excluded += 1
                continue
            m = re.match(r"^([0-9a-f]{64})\s+[\d,]+\s+(\S.*)$", line.rstrip("\n"))
            if not m:
                continue
            digest, rel = m.group(1), m.group(2).strip()
            if not os.path.isfile(rel):
                missing += 1
                problems.append("MISSING   " + rel)
            elif sha256_file(rel) == digest:
                ok += 1
            else:
                bad += 1
                problems.append("MISMATCH  " + rel)
    print("verified: %d  mismatch: %d  missing: %d  excluded(skipped): %d"
          % (ok, bad, missing, excluded))
    for p in problems[:10]:
        print("  " + p)
    if bad or missing:
        print("INTEGRITY-FAIL")
        return 1
    print("INTEGRITY-OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
