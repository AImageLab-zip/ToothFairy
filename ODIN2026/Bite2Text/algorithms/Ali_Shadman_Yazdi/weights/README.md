# Model weights

```
weights/
├── tsgcnet/
│   ├── upper.pth            16 MB   tooth segmentation, upper jaw   (in the repo)
│   └── lower.pth            16 MB   tooth segmentation, lower jaw   (in the repo)
├── photo_cnn/
│   ├── best.pt              43 MB   multi-view ResNet-18, 10 photo findings   (in the repo)
│   └── results.json                 dev-set scores + which heads are trusted
└── clik/                            CLIK crown-landmark detectors   (download)
    ├── [Crown]incisor-e837.pt      618 MB
    ├── [Crown]cuspid-e879.pt       618 MB
    ├── [Crown]premolar-e443.pt     618 MB
    └── [Crown]molar-e178.pt        618 MB
```

The CLIK checkpoints are not redistributed here. They belong to the
[CLIK-Diffusion](https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion) authors
and are downloaded from the Google Drive folder linked in their README:

```bash
python scripts/download_weights.py          # fetches the four [Crown]*.pt files, verifies SHA-256
python scripts/download_weights.py --check  # only verify what is on disk
```

Or download them by hand from
<https://drive.google.com/drive/folders/1o9tVJ6p8Jbad3gu0ZUkX0tE5dp7Vkh9g>
(only the four `[Crown]*.pt` files are needed; the root and diffusion
checkpoints are for CBCT teeth and tooth alignment) and place them in
`weights/clik/`.

To keep weights elsewhere, set `BITE2TEXT_WEIGHTS=/path/to/weights`. In the
Grand Challenge container this directory is mounted at `/opt/ml/model`.
