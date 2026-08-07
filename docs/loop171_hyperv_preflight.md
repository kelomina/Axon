# Loop171 Hyper-V Read-only Preflight

`scripts/Invoke-Loop171HyperVPreflight.ps1` is the fail-closed infrastructure
gate before the isolated Ubuntu recovery route. It is deliberately not a VM
provisioner or an experiment launcher. It only reads host state and writes a
JSON receipt to an already-existing report directory.

## What It Proves

The script passes only when all of these conditions are current at invocation:

- The Windows token is elevated.
- Available physical memory is at least `13 GiB` (`13958643712` bytes).
- The staged official Ubuntu Azure VHD archive is exactly
  `ubuntu-24.04-server-cloudimg-amd64-azure.vhd.tar.gz` and has SHA-256
  `05b7b5bb6172e5b0dd1248d5598c1bc27927c4625ba4c09c0442d4751725c43f`.
- The staged Linux guest asset is exactly `capa-v9.4.0-linux.zip` and has
  SHA-256 `07800a1d20a21eb18fc98716e2ae81b668e0c9a04defd588c8aa17ea3d3281e4`.
  This digest is the GitHub release-asset digest for Mandiant capa `v9.4.0`.
- The supplied dedicated root already exists, has no contents or reparse-point
  ancestors, uses a protected non-inheriting allowlist-only ACL, and is outside
  both staged archives and the receipt location.
- The planned VM name is unused, and the dedicated root has no intersection
  with current VM configuration, disk, checkpoint, smart-paging, or default
  Hyper-V storage paths. Existing switches are inventoried only; the preflight
  exposes no switch-selection parameter and records a required NIC count of
  zero.

The root ACL permits only `SYSTEM`, `BUILTIN\\Administrators`, and the current
elevated caller SID. It rejects inherited, deny, and non-allowlisted access
rules. Create it first with the dedicated provisioning script described below;
this read-only preflight will never create it.

## Protected-root Provisioning

`scripts/Initialize-Loop171ProtectedRoot.ps1` is the only Loop171 script that
creates directories. It requires an elevated token before it performs any
mutation, requires the requested dedicated parent and its root child to be new,
rejects reparse-point ancestors, creates any missing parent chain, and applies
the same protected allowlist ACL to every directory it creates. It records a
receipt outside the disposable parent/root, but does not create any report
directory itself.

The parent must itself be new. This keeps the disposable root below a dedicated
directory whose ACL is also protected, rather than changing ACLs on existing
user storage.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Initialize-Loop171ProtectedRoot.ps1 `
  -DedicatedParent D:\AxonLoop171Isolated `
  -RootDirectoryName root `
  -ReceiptPath E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop171\protected_root_creation.json
```

Only after this command returns
`protected_root_created_no_vm_or_sample_action_authorized` can the result be
passed to the read-only preflight. A failure is fail-closed and never permits
VM creation, VHD operations, mounting, guest boot, sample access, parsing,
training, heldout access, or F1 claims.

If setup fails after creating a directory, it attempts non-recursive cleanup in
reverse creation order. It deletes only directories recorded as created by that
invocation, and only when each is still an empty ordinary directory with no
reparse-point ancestor. A failed cleanup is an explicit
`protected_root_cleanup_incomplete_hard_error` in the receipt and requires
manual remediation; the script never uses recursive deletion.

## Invocation

Run from an elevated Windows PowerShell after both archives are complete and a
new empty protected root exists. The output parent must already exist and must
not be inside the disposable root.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Invoke-Loop171HyperVPreflight.ps1 `
  -BaseImageArchive E:\Project\python\Axon_v2.6Exp\.cache\loop171_hyperv\ubuntu_noble_20260713\ubuntu-24.04-server-cloudimg-amd64-azure.vhd.tar.gz `
  -LinuxCapaArchive E:\Project\python\Axon_v2.6Exp\.cache\loop171_hyperv\capa_linux_v9.4.0\capa-v9.4.0-linux.zip `
  -DedicatedRoot D:\AxonLoop171Isolated\root `
  -PlannedVmName AxonLoop171CapaIsolated `
  -ReceiptPath E:\Project\python\Axon_v2.6Exp\reports\roadmap_9997\loop171\hyperv_preflight.json
```

An exit code of `2` and `preflight_blocked_fail_closed` means no VM, VHD,
mount, guest, sample, parser, training, heldout access, or F1 claim is allowed.
A pass is only a prerequisite for the separately implemented harmless-fixture
acceptance run. It does not authorize sample access.

## Non-actions

The script does not contain VM, VHD, switch, mount, guest-service, sample, or
parser commands. It neither downloads nor extracts either archive. The Linux
capa ZIP is checked because an Ubuntu guest cannot use the existing Windows
`capa.exe`; extraction and guest-only execution remain deferred until after the
isolation acceptance evidence passes.
