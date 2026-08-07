# Loop171 Linux Guest Capa Adapter

`scripts/run_loop171_capa_guest.py` is a future Ubuntu-guest-only adapter for
the Linux `capa-v9.4.0-linux.zip` asset. It is static code and does not grant
sample access, training, held-out access, or an F1 claim.

The `install-toolchain` subcommand verifies the frozen archive SHA-256 and its
read-only mount,
rejects zip traversal, links, duplicate members, and oversized extraction, and
extracts only to a new guest-local directory. Its receipt binds the archive,
Linux ELF binary, and rules-tree digests. This setup action must occur only
after the harmless VM acceptance gate and before any authorized source is
presented to the guest.

The `run` subcommand is intentionally narrower than the former Windows worker:

- It accepts one future Train-only source only when its exact size and SHA-256
  match both before and after capa, every existing path component is not a
  symlink, and its Linux mount is `ro` according to `/proc/self/mountinfo`.
- It rejects a non-ELF capa executable and requires the previously fixed,
  read-only-mounted
  archive, binary, and rules-tree SHA-256 values before launch.
- It starts capa in its own Linux session, drains only stdout in memory with a
  `64 MiB` cap, discards stderr, and kills the process group on timeout or
  overflow. It never writes capa JSON to disk.
- It persists at most a `64 KiB` canonical receipt containing only rule counts
  and namespace counts. Source paths, source SHA-256 values, rule names, match
  locations, stderr, and raw capa JSON are absent from that receipt.

The guest's zero-NIC condition is proved by the separate Hyper-V harmless
acceptance receipt, not by this Python adapter. The adapter additionally gives
capa an offline proxy environment; this is defense in depth, not a substitute
for zero NIC isolation.

The exact static contract is
`manifests/roadmap_9997/loop171_hyperv_isolation/guest_capa_runner_contract.json`.
