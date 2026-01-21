import os, sys, time, re
from io import StringIO
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

base_url = "https://www.hockey-reference.com/leagues/NHL_{}_skaters.html"
years = range(2009, 2026)

sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
})

def uncomment(html):
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.replace_with(c)
    return str(soup)

def find_table_html(html):
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", id=re.compile(r"(stats|skaters|skaters_basic)"))
    return str(t) if t else None

def get_table(url):
    for _ in range(4):
        r = sess.get(url, timeout=30)
        if r.status_code in (429, 403):
            time.sleep(3)
            continue
        if r.status_code != 200:
            return None
        raw = r.text
        html = find_table_html(raw) or find_table_html(uncomment(raw))
        if not html:
            time.sleep(1.5)
            continue
        try:
            df = pd.read_html(StringIO(html), flavor="lxml")[0]
        except Exception:
            df = pd.read_html(StringIO(html))[0]
        return df
    return None

all_stats = []
for y in years:
    df = get_table(base_url.format(y))
    if df is None:
        print(f"Skipped {y} (no table found or blocked)", file=sys.stderr)
        continue
    df["Season"] = y
    all_stats.append(df)
    time.sleep(1.2)

if not all_stats:
    print("No data collected.", file=sys.stderr)
    sys.exit(1)

full_df = pd.concat(all_stats, ignore_index=True)
out = "../raw/hockey_reference_09-25.csv"
full_df.to_csv(out, index=False)
print(f"Wrote {out}", file=sys.stderr)
