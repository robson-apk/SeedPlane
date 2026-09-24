# SeedPlane cluster protocol v1

Status: experimental control plane for V25. It replaces the legacy Python-pickle transport for discovery, health and
coordination. Model execution is not enabled through v1 yet; the existing `serve` path remains available only for a
trusted LAN while the binary data plane is implemented.

## Start a worker

Use the same cluster ID and key on the coordinator and worker. Generate them once, store the key in a password manager,
and transfer them through a trusted channel:

```bash
export SEEDPLANE_CLUSTER_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export SEEDPLANE_CLUSTER_KEY="replace-with-at-least-32-random-characters"

# Worker machine. Loopback is the safe default.
seedplane worker

# A LAN bind is accepted only when SEEDPLANE_CLUSTER_KEY is explicit.
seedplane worker --host 0.0.0.0 --port 52100
```

On the coordinator:

```bash
export SEEDPLANE_CLUSTER_ID="the-same-uuid"
export SEEDPLANE_CLUSTER_KEY="the-same-key"
seedplane devices --address 192.168.1.20:52100
seedplane devices test --address 192.168.1.20:52100
seedplane doctor
```

Until mutual TLS/Noise pairing lands, expose the port only on a trusted private LAN, Tailscale or another authenticated
VPN. HMAC authenticates messages but does not encrypt their contents.

## Frame

Each control message is:

```text
4 bytes  magic = SPV1
4 bytes  unsigned big-endian payload length
N bytes  canonical UTF-8 JSON envelope
```

The envelope contains exactly `message` and `mac`. `mac` is HMAC-SHA256 over canonical JSON for `message`. Frames are
limited to 1 MiB and the declared length is rejected before payload allocation.

Required message fields:

```text
version, type, message_id, cluster_id, worker_id,
request_id, session_id, generation, deadline_ms, payload
```

Optional context fields are `model_hash`, `plan_hash` and `sent_ms`. IDs are canonical UUIDs; hashes are SHA-256 hex;
generations are non-negative integers; payload must be a JSON object. NaN/Infinity and unknown fields are rejected.

## Enabled operations

The initial worker allowlist is deliberately small:

- `hello`
- `capabilities`
- `health` / `heartbeat`
- `model_status`
- `cancel`

`load`, `benchmark`, `unload` and the binary execution plane are reserved but disabled until their resource limits,
cache policy and failure tests are complete. The worker never accepts a shell command.

## Failure behavior

- Invalid MAC: connection closes without a diagnostic oracle.
- Oversized frame: rejected before reading/allocating the payload.
- Expired deadline: rejected.
- Duplicate `message_id`: rejected.
- Older generation for the same `(session_id, request_id)`: rejected.
- Wrong `cluster_id`: rejected.
- Wrong model/plan hash: execution handlers must call `require_context` before work.
- Unknown field or operation: rejected.

The replay cache is bounded. Prompts are not logged by the protocol implementation.

## What remains for V25

- one-time pairing code and mutual TLS/Noise identities;
- persistent device registry and revocation (`devices forget`);
- binary data-plane header and checksums;
- cache transfer/load with size and disk quotas;
- cancellation connected to actual backend jobs;
- heartbeat leases, reconnect and fault injection;
- LAN bandwidth/latency measurements after the X79 link is repaired to 1 Gb/s.

## Tests

```bash
python -m unittest tests.test_cluster_protocol -v
```

The suite covers authenticated round trips, tampering, bad magic, oversized declarations, deadline expiry, invalid
hashes, replay, stale generations, cluster mismatch, disabled operations, CLI security defaults and a real loopback
worker probe.
