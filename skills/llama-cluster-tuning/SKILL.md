---
name: llama-cluster-tuning
description: Diagnose and tune a llama.cpp RPC cluster — choosing tensor-split proportions, spotting version mismatches, deciding whether distributing a model will help at all. Use when the user asks about multi-machine inference, slow generation across nodes, out-of-memory errors while loading, or rpc-server connection problems.
license: MIT
metadata:
  author: kestrel
  version: "1.0"
---

# Tuning a llama.cpp RPC cluster

## First question: should this be distributed at all?

RPC pools *memory*, not compute. If the model already fits on one machine,
adding workers makes generation slower, because every layer boundary that
crosses a machine now costs a network round trip. Distribute when the model does
not fit otherwise. Do not distribute for speed.

## Version mismatch is the usual culprit

The ggml wire format changes between releases. Symptoms are a hang during the
handshake, or a crash partway through loading. Every machine must run the same
llama.cpp build.

```
# on each machine
llama-server --version
rpc-server --version
```

Pin one tag everywhere and rebuild rather than mixing releases.

## Getting the split right

llama.cpp divides layers evenly by default, which strands memory whenever the
machines differ. Set proportions in device order — local devices first, then
each `--rpc` host in the order given:

```
llama-server -m model.gguf -ngl 99 \
  --rpc 192.168.1.10:50052,192.168.1.11:50052 \
  --tensor-split 0.5,0.3,0.2
```

Those are relative weights, not gigabytes. Match them to each machine's real
free memory. Run `scripts/split.py` with the memory figures to get the numbers.

## Reading the load log

A healthy start prints one buffer line per backend:

```
load_tensors: RPC[192.168.1.10:50052] model buffer size = 18022.14 MiB
load_tensors: RPC[192.168.1.11:50052] model buffer size = 11498.30 MiB
load_tensors:        CPU model buffer size =  9640.05 MiB
```

Three lines means three machines are holding weights. One line means the RPC
backends were not used — check that the workers are reachable and that the build
has `GGML_RPC=ON`.

## Worker flags that matter

- `-H 0.0.0.0` — the default binds to localhost, which is useless across machines.
- `-m <MB>` — the memory pool the worker advertises. Set it to real free memory.
- `-c` — local tensor cache. Costs disk, saves a lot of time on reload.

## Security

The RPC protocol has no authentication and no encryption. Keep it on a trusted
LAN or a VPN. Never expose a worker port to the internet.
