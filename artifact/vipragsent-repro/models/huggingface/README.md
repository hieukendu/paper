# Local model repositories

No local model weights are required for the no-fine-tuning experiment runner.
Fine-tuned/SFT systems should be run on a third-party platform and imported as
prediction JSONL files under `results/predictions/...`.

## Status

| Model | HF repo | Local path | Status | Size note |
| --- | --- | --- | --- | --- |
| PhoBERT-base | `vinai/phobert-base` | not present | Removed locally | Fine-tuning/inference should be done off-machine if needed. |
| XLM-R-large | `FacebookAI/xlm-roberta-large` | not present | Removed locally | Fine-tuning/inference should be done off-machine if needed. |
| Sailor-7B | `sail/Sailor-7B` | not present | Removed locally | 7B hosting is GPU-heavy; import predictions instead. |
| Vistral-7B | `Viet-Mistral/Vistral-7B-Chat` | not cloned | Blocked by Hugging Face gated access | HF API reports 29,179,398,902 bytes of used storage. The current token is not authorized for git access. |

## Re-download commands

Run these only if you deliberately want to host or fine-tune locally.

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
