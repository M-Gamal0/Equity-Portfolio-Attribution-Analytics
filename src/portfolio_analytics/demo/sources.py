"""Cached public Nasdaq/Yahoo/French sources with explicit offline replay."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

UNIVERSE_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FACTOR_URLS = {
    "ff5": FRENCH_BASE + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": FRENCH_BASE + "F-F_Momentum_Factor_daily_CSV.zip",
}
FACTOR_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SourceCache:
    """One frozen acquisition directory. Offline reads verify each source hash."""

    def __init__(self, folder, *, offline=False):
        self.folder = Path(folder)
        self.offline = offline
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, name, source, fetch):
        path = self.folder / name
        meta_path = path.with_name(path.name + ".source.json")
        if path.exists() or meta_path.exists():
            if not path.is_file() or not meta_path.is_file():
                raise ValueError(f"Incomplete cached source: {name}")
            data, meta = path.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["source"] != source or meta["sha256"] != sha256(data).hexdigest():
                raise ValueError(f"Cached source provenance/hash mismatch: {name}")
            return data
        if self.offline:
            raise FileNotFoundError(f"Offline cache is missing {name}; complete acquisition online first.")
        data = fetch()
        if not isinstance(data, bytes) or not data:
            raise ValueError(f"Empty public source: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Metadata is the commit marker; partial writes are never reused as valid.
        path.write_bytes(data)
        meta_path.write_text(json.dumps({"source": source, "retrieved_at_utc": utc_now(),
                                        "sha256": sha256(data).hexdigest(), "bytes": len(data)}, indent=2), encoding="utf-8")
        return data

    def provenance(self):
        records = {}
        for path in sorted(self.folder.rglob("*.source.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            source_file = path.with_name(path.name.removesuffix(".source.json"))
            if not source_file.is_file() or sha256(source_file.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Source cache checksum mismatch: {source_file.name}")
            records[path.relative_to(self.folder).as_posix()] = metadata
        return records


def _request(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 (portfolio-analytics public research demo)",
                                            "Accept": "application/json,text/plain,*/*"})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def parse_universe(data):
    payload = json.loads(data)
    rows = payload["data"]["data"]["rows"]
    frame = pd.DataFrame(rows)[["symbol", "companyName"]].rename(
        columns={"symbol": "ticker", "companyName": "company_name"})
    frame["ticker"] = frame.ticker.astype(str).str.strip().str.replace(".", "-", regex=False)
    if len(frame) < 100 or frame.ticker.duplicated().any() or not frame.ticker.str.fullmatch(r"[A-Z0-9^-]+").all():
        raise ValueError("Unexpected Nasdaq-100 universe schema/membership; inspect the cached source.")
    return frame.sort_values("ticker", kind="stable").reset_index(drop=True)


def fetch_universe(cache):
    return parse_universe(cache.get("nasdaq100.json", UNIVERSE_URL, lambda: _request(UNIVERSE_URL)))


def _yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install the demo extra: pip install "portfolio-analytics[demo]"') from exc
    return yf


def fetch_quote(cache, ticker):
    def download():
        info = _yfinance().Ticker(ticker).get_info()
        fields = ("marketCap", "sector", "currency", "longName", "quoteType", "regularMarketTime")
        selected = {name: info.get(name) for name in fields}
        selected["ticker"] = ticker
        return json.dumps(selected, sort_keys=True, allow_nan=False).encode()
    return json.loads(cache.get(f"quotes/{ticker}.json", f"yfinance:Ticker.get_info:{ticker}", download))


def fetch_prices(cache, ticker, start, end):
    """Explicit auto_adjust=False preserves Close and dividend-adjusted Adj Close."""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    source = f"yfinance:history:{ticker}:{start}:{end_exclusive}:auto_adjust=False:repair=False"

    def download():
        frame = _yfinance().Ticker(ticker).history(start=start, end=end_exclusive, interval="1d",
                                                 auto_adjust=False, actions=False, repair=False,
                                                 keepna=True, timeout=30)
        if frame.empty or not {"Close", "Adj Close"}.issubset(frame):
            raise ValueError(f"Yahoo has no usable unadjusted/adjusted history for {ticker}")
        frame = frame[["Close", "Adj Close"]].copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index.name = "date"
        stream = BytesIO()
        frame.reset_index().to_parquet(stream, index=False)
        return stream.getvalue()

    content = cache.get(f"prices/{ticker}_{start}_{end}.parquet", source, download)
    frame = pd.read_parquet(BytesIO(content)).set_index("date").sort_index()
    if frame.index.has_duplicates or not {"Close", "Adj Close"}.issubset(frame):
        raise ValueError(f"Duplicate dates or malformed history: {ticker}")
    return frame


def parse_french_zip(content, *, momentum=False):
    with ZipFile(BytesIO(content)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one CSV in French archive")
        text = archive.read(members[0]).decode("utf-8-sig")
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if ("Mom" in line if momentum else "Mkt-RF" in line) and "," in line), None)
    if header is None:
        raise ValueError("French factor CSV header not found")
    # Only daily YYYYMMDD rows; exclude annual/monthly sections and prose.
    daily = [line for line in lines[header + 1:] if re.match(r"^\s*\d{8}\s*,", line)]
    frame = pd.read_csv(StringIO(lines[header] + "\n" + "\n".join(daily)))
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.rename(columns={frame.columns[0]: "date", "Mkt-RF": "mkt_rf", "SMB": "smb",
                                  "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf", "Mom": "mom"})
    columns = ["mom"] if momentum else ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    frame["date"] = pd.to_datetime(frame.date.astype(str), format="%Y%m%d", errors="raise")
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    if frame.empty or frame.date.duplicated().any() or not np.isfinite(values.to_numpy()).all() or values.le(-99).any().any():
        raise ValueError("Invalid daily French factors or missing-value sentinel")
    frame[columns] = values / 100.0
    return frame[["date", *columns]]


def fetch_factors(cache, start, end):
    frames = [parse_french_zip(cache.get(name + ".zip", url, lambda url=url: _request(url)),
                               momentum=name == "momentum") for name, url in FACTOR_URLS.items()]
    frame = frames[0].merge(frames[1], on="date", how="inner", validate="one_to_one")
    return frame.loc[frame.date.between(start, end)].sort_values("date").reset_index(drop=True)
