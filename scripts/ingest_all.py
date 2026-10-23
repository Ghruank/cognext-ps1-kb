import json, time, pickle
from engine.ingest import parse_folder
t = time.time()
docs = parse_folder("data/Documents", workers=10)
print(f"{sum(d['n_pages'] for d in docs)} pages, {sum(len(d['units']) for d in docs)} units, "
      f"{sum(d['ocr_pages'] for d in docs)} OCR pages in {time.time()-t:.0f}s")
pickle.dump(docs, open("storage/train_docs.pkl", "wb"))
