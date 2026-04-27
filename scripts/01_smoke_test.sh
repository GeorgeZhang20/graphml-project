#!/usr/bin/env bash
# Quick smoke test: train a 2-layer GCN on Cora using PyG. ~3 seconds on CPU.
# Confirms env is working before touching TAPE.
set -euo pipefail
python - <<'PY'
import time, torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

ds = Planetoid(root="TAPE/dataset/Cora", name="Cora")
data = ds[0]

class GCN(torch.nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = GCNConv(i, h), GCNConv(h, o)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, self.training)
        return self.c2(x, ei)

torch.manual_seed(0)
m = GCN(ds.num_features, 16, ds.num_classes)
opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
t0 = time.time()
for _ in range(200):
    m.train(); opt.zero_grad()
    F.cross_entropy(m(data.x, data.edge_index)[data.train_mask],
                    data.y[data.train_mask]).backward()
    opt.step()
m.eval()
acc = (m(data.x, data.edge_index).argmax(1)[data.test_mask]
       == data.y[data.test_mask]).float().mean().item()
print(f"Cora GCN: test_acc={acc:.4f} in {time.time()-t0:.1f}s")
assert acc > 0.78, "smoke test failed"
print("OK")
PY
