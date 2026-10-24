"""Parse a folder of PDFs with neo.ingest and save the units (no LLM).

    python -m alpha.ingest_all data/Documents train
"""

import pickle
import sys
import time

from alpha.ingest import parse_folder
from alpha.pipeline import STORE

folder, collection = sys.argv[1], sys.argv[2]
t = time.time()
docs = parse_folder(folder)
print(f"{len(docs)} docs, {sum(d['n_pages'] for d in docs)} pages, {sum(len(d['units']) for d in docs)} units, "
      f"{sum(d['ocr_pages'] for d in docs)} OCR pages in {time.time() - t:.0f}s")
pickle.dump(docs, open(STORE / f"{collection}_docs.pkl", "wb"))
