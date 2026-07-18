# Local model repositories

No local model weights are required for the no-fine-tuning experiment runner.
Fine-tuned/SFT systems should be run on a third-party platform and imported as
prediction JSONL files under `results/predictions/...`.

## Status

| Model | HF repo | Pinned upstream revision | Local path | Status | Size note |
| --- | --- | --- | --- | --- |
| PhoBERT-base | `vinai/phobert-base` | not pinned | not present | Removed locally | Fine-tuning/inference should be done off-machine if needed. |
| XLM-R-large | `FacebookAI/xlm-roberta-large` | not pinned | not present | Removed locally | Fine-tuning/inference should be done off-machine if needed. |
| Sailor-7B | `sail/Sailor-7B` | `b8b49a0f02073e58db2e42e5811955dfe87ca970` | not present | Removed locally | Resolved from upstream HEAD on 2026-07-18; this is not retrospective proof for the historical run. |
| Vistral-7B | `Viet-Mistral/Vistral-7B-Chat` | `d331b64e61b935cc43c2b3010ae9fb4fde599b45` | not cloned | Blocked by Hugging Face gated access | Resolved from upstream HEAD on 2026-07-18; this is not retrospective proof for the historical run. |

## Re-download commands

Run these only if you deliberately want to host or fine-tune locally.
`scripts/16_download_models.py` retrieves Sailor and Vistral at the pinned
revisions in `configs/models.yaml`.

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://huggingface.co/vinai/phobert-base models\huggingface\vinai__phobert-base
git clone https://huggingface.co/FacebookAI/xlm-roberta-large models\huggingface\FacebookAI__xlm-roberta-large
git clone https://huggingface.co/sail/Sailor-7B models\huggingface\sail__Sailor-7B
```

For Vistral, first request/accept access on Hugging Face for
`Viet-Mistral/Vistral-7B-Chat`, then clone with LFS skipped and pull the
safetensor shards:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --depth 1 https://huggingface.co/Viet-Mistral/Vistral-7B-Chat models\huggingface\Viet-Mistral__Vistral-7B-Chat
cd models\huggingface\Viet-Mistral__Vistral-7B-Chat
git lfs pull --include="model-*.safetensors,model.safetensors.index.json" --exclude=""
```
