# My fine-tuned ResNet-18

The website shows this model next to the frozen SwinV2. Its weights are in
`models/jpeg/best.pt` and its settings (cut-off 0.61) in `models/jpeg/card.json`.
`modal_app.py` loads both on deploy, so nothing here needs to run for the site.

## Training again (optional)

This downloads the public `Thilak5434/AI-vs-Real` set and takes about an hour on a CPU.

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
set KMP_DUPLICATE_LIB_OK=TRUE
python prepare_data.py
python train.py --run jpeg
python compare_swin.py
python evaluate.py
```

| File | What it does |
| --- | --- |
| `prepare_data.py` | Downloads the dataset and writes `data/manifest.csv` |
| `train.py` | Fine-tunes ResNet-18 (ImageNet start) and saves `models/<run>/best.pt` |
| `compare_swin.py` | Scores the same test set with the frozen SwinV2 |
| `evaluate.py` | Writes test metrics and charts to `results/` |
| `common.py` | Shared model, transforms and metrics |
| `samples/` | One AI picture and one real photo for quick tests |
