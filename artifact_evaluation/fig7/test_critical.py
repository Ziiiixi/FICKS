#!/usr/bin/env python3
import re
import pandas as pd

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
LOG_FILE    = "123.log"
OUTPUT_CSV  = "colocations.csv"
# ───────────────────────────────────────────────────────────────────────────────

def main():
    pair_re = re.compile(r"will colocation\s+(\d+)\s+and\s+(\d+)")
    dur_re  = re.compile(r"Colocation duration_nano:\s*([\d\.]+)")

    rows = []
    with open(LOG_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        m_pair = pair_re.match(lines[i])
        if m_pair and i + 1 < len(lines):
            m_dur = dur_re.match(lines[i+1])
            if m_dur:
                id1, id2 = int(m_pair.group(1)), int(m_pair.group(2))
                duration = float(m_dur.group(1))
                rows.append({'id1': id1,
                             'id2': id2,
                             'duration_nano': duration})
                i += 2
                continue
        i += 1

    # write out with pandas
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
