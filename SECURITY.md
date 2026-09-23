# Security policy

## Supported version

SeedPlane is alpha research software. Security fixes are applied to the latest commit on `main`.

## Network safety

Python workers use `multiprocessing.connection`, whose payload is pickle. The native worker uses a binary TCP protocol.
Neither transport encrypts traffic. Both bind to `127.0.0.1` by default and refuse non-loopback binds unless
`SEEDPLANE_AUTHKEY` is explicitly set.

- Never expose worker or llama.cpp RPC ports to the public internet.
- Prefer loopback, an SSH tunnel, or a private VPN.
- On a trusted LAN, use a long random `SEEDPLANE_AUTHKEY` shared out of band.
- Treat anyone who knows or can observe that key as able to control the worker.
- llama.cpp's RPC server has its own security model; isolate it from untrusted networks.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. If that option is unavailable, open an issue
that requests a private contact channel; do not include exploit details or secrets in a public issue.
