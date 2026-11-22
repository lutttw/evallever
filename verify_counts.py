# -*- coding: utf-8 -*-
"""verify_counts.py — scripted count reconciliation for the EvalLever bundle.

The manuscript's Data-availability sentence quotes a manifest count
("the sha256 manifest verifies N/N files"). Hand-typed counts drift; this
script is the single source of truth: it counts MANIFEST.txt entries
itself and, given manuscript/letter text files, checks every "N/N files"
or "N sha256-verified" claim against the machine count.

Usage:
  python verify_counts.py                 # print the machine count
  python verify_counts.py --check a.txt b.txt ...
Exit: 0 = all claims match; 1 = any mismatch ( prints the offenders ).
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def manifest_count():
    n = 0
    with io.open(os.path.join(HERE, "MANIFEST.txt"), encoding="utf-8") as f:
        for line in f:
            if re.match(r"^[0-9a-f]{64}\s+\d", line):
                n += 1
    return n

def main():
    mc = manifest_count()
    if "--check" not in sys.argv:
        print("manifest entries: %d" % mc)
        print('claim wording should read: "the sha256 manifest verifies %d/%d files"' % (mc, mc))
        return 0
    files = [a for a in sys.argv[1:] if a != "--check"]
    bad = []
    for path in files:
        try:
            t = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            bad.append((path, "unreadable: %s" % e))
            continue
        for m in re.finditer(r"(\d+)\s*(?:sha256-verified|/)\s*(\d+)?\s*files", t):
            nums = [int(m.group(1))] + ([int(m.group(2))] if m.group(2) else [])
            if any(x != mc for x in nums):
                bad.append((path, m.group(0)))
    if bad:
        print("MISMATCH (machine count = %d):" % mc)
        for p, s in bad:
            print("  %s: %s" % (p, s))
        return 1
    print("all file-count claims match the manifest (%d)." % mc)
    return 0

if __name__ == "__main__":
    sys.exit(main())
