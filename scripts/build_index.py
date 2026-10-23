import pickle, time
from engine.index import Index
docs = pickle.load(open("storage/train_docs.pkl", "rb"))
units = [u for d in docs for u in d["units"]]
t = time.time()
idx = Index(units).build()
print(f"{len(units)} units indexed in {time.time()-t:.0f}s, vecs {idx.vecs.shape}")
idx.save("storage/train_index.pkl")
