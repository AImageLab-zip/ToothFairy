# Release and privacy policy

Actual final2 code, calibrated parameters and checkpoint files are public under
the terms in `NOTICE.md`. Weights belong in versioned release assets, not Git
history. Every checkpoint must match `release/weights.json`.

Do not publish patient photographs, scans, reports, predictions, case identifiers,
training examples, credentials, private service URLs or notebook outputs. Public
calibration files contain inference parameters, not case-level evaluation records.
The kernel-disposition PLY file in the vendored 3DTeethLand runtime is a synthetic
convolution kernel, not a patient tooth mesh.

Checkpoint formats may contain Python pickle objects. Load only trusted artifacts
whose hashes match the release manifest. The downloader checks hashes before
extraction, rejects traversal and link entries, and does not deserialize weights.

Keep inference inputs/outputs outside Git. For accidental publication, contact
the repository owner, revoke exposed credentials and remove affected release
assets promptly. Review Git history as well as the working tree.
