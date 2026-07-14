# HUMAN_ACTIONS.md

## Trạng thái dữ liệu hiện tại - 2026-06-28

- [x] Secret đã được chuyển sang file `.env` bị git ignore; `.env.example` đã quay lại dạng mẫu, không chứa key thật.
- [x] Đã nạp bản export ViSoBERT cục bộ từ `D:\hf_cache\exports\visobert_12000_api`.
- [x] Đã tải/chuyển đổi các bộ dữ liệu công khai trên máy:
  - `data/raw/uit_vsfc/` -> 16.175 bản ghi, giữ nguyên split gốc.
  - `data/raw/uit_vsmec/` -> 6.927 bản ghi, giữ nguyên split gốc.
  - `data/raw/aivivn_2019/` -> 16.087 bản ghi từ Kaggle mirror.
- [x] Các file dữ liệu đã hợp nhất:
  - `data/interim/cleaned_social.jsonl` -> 12.000 bình luận social đã làm sạch.
  - `data/interim/agent_prelabeled.jsonl` -> 12.000 nhãn bạc do agent local/rule-based gán cho dữ liệu social.
  - `data/interim/public_datasets_unified.jsonl` -> 39.189 bản ghi từ các bộ public, đã đưa về schema chung.
  - `data/processed/all_unified.jsonl` -> 51.189 bản ghi tổng cộng.
- [x] Đã tạo batch để annotate trong `data/annotation/batches/` -> 120 batch, mỗi batch 100 bình luận social.
- [x] Đã tạo hàng đợi enrichment tại `data/annotation/enrichment_queue.jsonl` -> 1.808 bản ghi được chọn theo cue để ưu tiên rà nhãn hiếm.
- [x] Đã ghi report dữ liệu và checksum:
  - `data/manifest/data_status_report.json`
  - `data/manifest/datasets.json`
  - `data/manifest/checksums.json`
- [x] Đã ghi silver label cho toàn bộ annotation batch trong `data/annotation/agent_silver/` bằng đường local reasoning agent.
- [x] Đã ghi index silver label tại `data/manifest/agent_silver_summary.json` để liệt kê đủ 120 file batch đã được agent gán nhãn.
- [x] Đã thêm field `type` vào các file JSONL để giữ split bucket nguồn phục vụ chia train/dev/test sau này:
  - Public datasets có split nguồn thì `type` là `train`, `dev` hoặc `test`.
  - `visobert_local` giữ `type=null` vì 12.000 comment social sẽ được chia sau khi có human annotation/adjudication.
  - Các file batch input và agent silver hiện đã giữ đủ `source`, `split`, `type` và `platform`.
- [x] Đã ghi report type tại `data/manifest/source_type_report.json`.

Tổng phân bố `type` hiện tại trong `data/processed/all_unified.jsonl`:

- `type=train`: 28.557 bản ghi public.
- `type=dev`: 3.556 bản ghi public.
- `type=test`: 7.076 bản ghi public.
- `type=null`: 12.000 bản ghi `visobert_local`.

AIVIVN hiện được xử lý như sau:

- Nguồn Kaggle có `train.csv` và `test.csv`, không có dev gốc.
- Repo tách deterministic 10% từ `train.csv` làm dev bằng seed `20260520`.
- Sau khi tách: `train=11.583`, `dev=1.287`, `test=3.217`.
- Các sample AIVIVN `type=dev` đều có nguồn từ `train.csv`; chi tiết được đánh dấu trong `provenance.source_type_note`.

Lưu ý quan trọng: các nhãn pragmatic sentiment trên dữ liệu social hiện mới là `silver_agent`. Đây chưa phải nhãn vàng/gold label, và không nên dùng làm nhãn train/dev/test cuối cùng của ViPragSent trước khi có người rà soát và adjudication.

## Việc còn thiếu / cần con người suy luận và xác nhận

- [ ] Xác nhận license của UIT-VSFC, UIT-VSMEC, AIVIVN và ViSoBERT trước khi release raw text hoặc viết claim trong báo cáo/paper.
- [ ] Xác nhận nguồn AIVIVN: Kaggle mirror hiện có ở đây chỉ là binary positive/negative và chỉ có train/test. Repo đã tách 10% từ `train.csv` làm `type=dev`, nhưng điều này vẫn không khớp với workflow đang kỳ vọng AIVIVN dạng 3 nhãn/3-way split.
- [ ] Quyết định sẽ cung cấp đúng nguồn AIVIVN 3-way như paper dùng, hay giữ Kaggle binary mirror hiện tại và ghi caveat rõ ràng.
- [ ] Rà file `data/interim/agent_prelabeled.jsonl`; đây là nhãn bạc rule-based/agent local, chưa phải nhãn LLM trả phí và cũng chưa phải nhãn vàng do người annotate.
- [ ] Phân công hai reviewer cho `data/annotation/batches/`, sau đó lưu file reviewer vào:
  - `data/annotation/reviewer_01/`
  - `data/annotation/reviewer_02/`
- [ ] Chạy `scripts/05_merge_human_annotations.py` sau khi có file từ reviewer, rồi xử lý bất đồng trong `data/annotation/disagreements/`.
- [ ] Cần enrichment cho các nhãn hiếm: sarcasm, irony, idiom/figurative, code-switching và implicit sentiment hiện vẫn có số lượng silver thấp. Nên bắt đầu từ `data/annotation/enrichment_queue.jsonl`.
- [ ] Chưa crawl trực tiếp Facebook/TikTok/YouTube. Giữ `ENABLE_SOCIAL_CRAWL=false` cho đến khi đã rà ToS/pháp lý/IRB.

## 1. Secret/API

- [x] Copy `.env.example` thành `.env`.
- [x] Điền `HF_TOKEN`, `KAGGLE_USERNAME` và `KAGGLE_KEY`.
- [x] Xác nhận `.env` đã bị git ignore.
- [ ] Chỉ điền/xác nhận biến Azure OpenAI nếu muốn dùng LLM trả phí để silver-label hoặc sinh rationale. Hiện không bắt buộc vì mặc định đang dùng local heuristic labeling:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_API_VERSION`
  - `AZURE_OPENAI_DEPLOYMENT_LABEL`
  - `AZURE_OPENAI_DEPLOYMENT_RATIONALE`

## 2. Dataset/License

- [x] Xác nhận path ViSoBERT cục bộ tồn tại: `D:\hf_cache\exports\visobert_12000_api`.
- [ ] Xác nhận quyền truy cập và license của UIT-VSFC.
- [ ] Xác nhận quyền truy cập/license của UIT-VSMEC. Mapping hiện đã chuẩn hóa theo 7 nhãn UIT-VSMEC: `enjoyment`, `sadness`, `anger`, `disgust`, `fear`, `surprise`, `other`.
- [ ] Xác nhận nguồn và chính sách split của AIVIVN-2019, đặc biệt vì bản Kaggle đã tải chỉ là binary/train-test; dev hiện là phần tách từ train với seed `20260520`.
- [ ] Xác nhận raw text có được lưu cục bộ hoặc release lại dưới phạm vi research-only hay không.
- [ ] Xác nhận không bật crawl trực tiếp mạng xã hội nếu chưa rà ToS/pháp lý.

## 3. Human Annotation/Adjudication

- [ ] Rà nhãn bạc do agent tạo cho toàn bộ batch, bắt đầu từ `batch_001`.
- [ ] Phân công `reviewer_01` và `reviewer_02`.
- [ ] Xử lý disagreement trong `data/annotation/disagreements/`.
- [ ] Duyệt nhãn đã adjudicate trước khi tạo JSONL processed cuối cùng.

## 4. Rationale Audit

- [ ] Audit ít nhất 5% rationale đã sinh.
- [ ] Loại bỏ rationale không faithful/không đúng với quyết định nhãn.

## 5. Fine-Tuning đang để sau

- [ ] Không chạy training trên Quadro P2000 local trừ khi đã chấp nhận rõ ràng.
- [ ] Chuyển full Q1-Q4 lên cloud GPU nếu muốn chạy đúng thí nghiệm.
- [ ] Chỉ set `ENABLE_FINE_TUNING=true` sau khi dữ liệu/adjudication đã được duyệt.
- [ ] Chỉ truyền `--confirm-run-training` cho các lần training đã được phê duyệt.
