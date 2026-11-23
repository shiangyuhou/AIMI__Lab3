# AIMI__Lab3
## GitHub Link

[GitHub：shiangyuhou/AIMI__Lab3](https://github.com/shiangyuhou/AIMI__Lab3)

### 繳交檔案

`submission_best.csv` 是 Kaggle 公榜得分 **0.88168** 的提交檔，共 1,182 筆預測。

### 建立執行環境

進入 `train_cxr_hybrid.py` 所在的目錄。實測使用 Python 3.10.12，並可用 `nvidia-smi` 確認 NVIDIA GPU。若 Ubuntu 22.04 尚未安裝 `venv`，先執行 `sudo apt install python3.10-venv`。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==5.20.2 numpy==2.2.6 pillow==12.3.0 tqdm==4.70.1 \
  huggingface-hub==2.4.0 tokenizers==0.23.3 safetensors==0.8.0
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
```

### 從頭重訓

將課程提供的 `nycu-aimi-2026-lab3-cxr.zip` 和 `test_data_sample.csv` 放入程式同層的 `data/`。首次訓練若沒有快取，程式會下載 RAD-DINO 和 ConvNeXt-Tiny 的官方預訓練參數。重訓預設使用 seed 42 訓練六輪，依驗證 Macro F1 選擇最佳輪次。

```bash
python train_cxr_hybrid.py --output-dir retrain_output
cmp submission_best.csv retrain_output/submission.csv
```

實測重訓產生的提交檔與 `submission_best.csv` 逐位元組一致。`cmp` 沒有輸出即表示兩份檔案相同。

有機會，因為裝置不同而與最終提交檔案略有出路。
