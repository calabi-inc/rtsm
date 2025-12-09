from rtsm.stores.vectors.faiss_client import FaissClient
import numpy as np

cfg = {"vectors": {"enable": True, "backend": "faiss", "dim": 512}}
vc = FaissClient(cfg)
a = np.random.randn(512).astype("float32"); a /= np.linalg.norm(a) + 1e-12
b = a.copy()
c = np.random.randn(512).astype("float32"); c /= np.linalg.norm(c) + 1e-12
vc.upsert_batch([{"object_id": "a", "emb": a}, {"object_id": "c", "emb": c}])
print(vc.search(b, top_k=2))  # should return [('a', ~1.0), ('c', lower score)]