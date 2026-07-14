# Kế Hoạch Kỹ Thuật: ViPragSent (08_vipragsent_emnlp)

Kế hoạch này biến gói bản thảo ViPragSent được tạo tự động thành một bài nộp
EMNLP/ARR thực sự, có thể bảo vệ được. Mọi tham chiếu tệp đều tính tương đối so
với thư mục gốc của dự án (`08_vipragsent_emnlp/`).

---

## 1. Tóm tắt bài báo & trạng thái hiện tại

**ViPragSent** ("Vietnamese Pragmatic Sentiment" — Phân tích cảm xúc dụng học tiếng Việt)
hướng đến các hiện tượng dụng học (pragmatic phenomena) trong mạng xã hội tiếng Việt
mà các benchmark phân cực bề mặt (surface-polarity) chưa bao phủ đầy đủ: cảm xúc
ngầm ẩn (implicit sentiment), mỉa mai (sarcasm), trào phúng (irony), thành ngữ/ngôn
ngữ bóng bẩy (idiom/figurative language), chuyển mã (code-switching), và giễu nhại
(mocking). Phương pháp (xem [`sections/03_method.tex`](sections/03_method.tex)) mở
rộng khoảng 8.4k bình luận Facebook/TikTok/YouTube đã được phân xét với sáu nhãn
dụng học nhị phân cộng với phân cực ý định (intended polarity) 3 chiều và cảm xúc
UIT-VSMEC-7, sau đó huấn luyện các backbone (PhoBERT-base, XLM-R-large, Vistral-7B,
Sailor-7B) dùng chung dưới hàm mất mát đa nhiệm vụ (multi-task loss) với trọng số
bất định đồng phương sai (homoscedastic-uncertainty weighting) cộng với một đầu
rationale chuỗi suy luận (chain-of-thought rationale head) chỉ dùng lúc huấn luyện,
được chưng cất từ GPT-4o-mini. Các khẳng định chính: +5.5 macro-pragmatic F1 so với
PhoBERT fine-tune (+7.4 sarcasm, +5.9 implicit), +3.3 so với GPT-4o-mini 8-shot, giữ
nguyên cảm xúc thông thường (+2.4 UIT-VSMEC), và ECE giảm một nửa (0.094 → 0.048).

**Cảnh báo trạng thái nghiêm trọng: mọi kết quả định lượng, hình vẽ, bảng, và bảng
phân tích định tính trong gói này đều là TỔNG HỢP (SYNTHETIC).** Tất cả các con số
được tạo ra bởi một bộ mô phỏng số (numerical simulator) với seed cố định
(`20260520`) tại [`scripts/make_artifacts.py`](scripts/make_artifacts.py) — điểm
cơ sở hardcode (`BASE_PHEN`, `ORD_BASE`), delta cộng (`SYS_DELTA`, `ORD_DELTA`),
danh sách ablation viết tay (`ABLATIONS`), đường cong logistic cho lát cắt tài nguyên
thấp (low-resource slice), ma trận nhầm lẫn (confusion matrix) hardcode, và các
"seed" được lấy mẫu Gaussian. Chưa có bộ dữ liệu nào được thu thập hay chú thích,
chưa có mô hình nào được huấn luyện, chưa có lời gọi API nào được thực hiện. Các
cấu hình ([`configs/*.yaml`](configs/)) chỉ là stub giữ chỗ (`primary_backbone: TBD`,
`datasets: pending_manifest`, `seeds: [1, 2, 3]` — không nhất quán với 5 seed mà bài
báo khẳng định). Các bảng trong `sections/*.tex` được hardcode nội tuyến và có thể
bị lệch so với `results/*.json`. **Tất cả các số và hình vẽ phải được thay thế bằng
kết quả thí nghiệm thực trước bất kỳ lần nộp bài nào.**
[`notes/completion_note.md`](notes/completion_note.md) xác nhận điều này: "Replace
every `Synthetic example result` value with real experiments."

---

## 2. Danh mục bảng đầy đủ

Năm bảng LaTeX. Không có bảng nào được tạo tự động; tất cả đều được hardcode nội
tuyến và phải được tạo lại từ `results/*.json` thực.

| # | `\label` | Nguồn (tệp:dòng) | Các chỉ số / cột được báo cáo | Thí nghiệm & tệp kết quả thực cần có |
|---|----------|--------------------|-------------------------------|----------------------------------------|
| T1 | `tab:datasets` | [`sections/04_experimental_setup.tex:19-36`](sections/04_experimental_setup.tex) | Bộ dữ liệu, Nhiệm vụ, Miền, Nhãn, Phân chia train/dev/test, Giấy phép. Các hàng: UIT-VSFC (11426/1583/3166), UIT-VSMEC (5548/686/693), AIVIVN-2019 (16087/1788/5454), ViPragSent (8412/2104/2106) | Thu thập dữ liệu & chú thích thực. Số lượng/giấy phép từ `data/manifest/datasets.json` thực (hiện tại là tổng hợp). Phải xác minh số lượng phân chia khớp với dữ liệu thực sự đã tải về/chú thích. |
| T2 | `tab:main_pragmatic` | [`sections/05_results.tex:28-58`](sections/05_results.tex) | Binary macro-F1 theo hiện tượng (Implicit, Sarcasm, Irony, Idiom, Code-sw., Mocking, Macro-prag) ±95% CI, cho 11 hệ thống (PhoBERT-ST, PhoBERT FT, XLM-R, Sailor-7B SFT, Vistral-7B SFT, GPT-4o-mini zero/8-shot, ViPragSent −noaux/−CoT-only/−exp-only, ViPragSent) | Huấn luyện cả 11 hệ thống trên tập train ViPragSent thực; đánh giá trên tập test 2.106 bình luận thực; 5 seed; CI paired-bootstrap. Tạo ra [`results/main_pragmatic.json`](results/main_pragmatic.json). |
| T3 | `tab:ordinary` | [`sections/05_results.tex:81-104`](sections/05_results.tex) | Macro-F1 trên UIT-VSFC, UIT-VSMEC, AIVIVN cho 8 hệ thống (không hiển thị CI) | Đánh giá từng hệ thống đã huấn luyện trên ba tập test công khai. Tạo ra [`results/ordinary_sentiment.json`](results/ordinary_sentiment.json). Bổ sung CI để khớp với độ nghiêm ngặt của T2. |
| T4 | `tab:ablation` | [`sections/05_results.tex:118-147`](sections/05_results.tex) | Prag. F1, Ord. F1, ECE (×10³), Chi phí tương đối (relative Cost), cho 8 cấu hình ablation (full, −emotion, −ord-sent, −expl/CoT, −multitask, −uncertainty-weighting, expl-augmented-inference, hard-label-distill) | Quét ablation một backbone đơn có kiểm soát (tập dev). Tạo ra [`results/multitask_ablation.json`](results/multitask_ablation.json). Lưu ý: chú thích nói "tập dev" và "backbone khác so với Bảng 2, do đó giá trị tuyệt đối cao hơn" — giữ lại sự phân biệt đó. |
| T5 | `tab:cost` | [`sections/05_results.tex:240-262`](sections/05_results.tex) | GPU-h, Chi phí tính toán (Compute $), Chi phí API (API $), Tổng $ mỗi hệ thống (8 hàng). Chi phí chú thích $1860 chiếm phần lớn | Hạch toán tính toán/API thực được ghi lại trong quá trình chạy T2/T4. Tạo ra [`results/cost_breakdown.json`](results/cost_breakdown.json). |

---

## 3. Danh mục hình vẽ đầy đủ

Tám hình vẽ. Hình 1 và 8 là sơ đồ vẽ tay (không có dữ liệu kết quả); Hình 2–7 vẽ
đồ thị từ mảng mô phỏng. Tất cả được tạo bởi các hàm vẽ trong
[`scripts/make_artifacts.py`](scripts/make_artifacts.py).

| # | `\label` | Hàm tạo (make_artifacts.py) | Nội dung vẽ | Nguồn dữ liệu thực cần có |
|---|----------|-----------------------------|-------------|---------------------------|
| F1 | `fig:pipeline` ([`05_results.tex:23`](sections/05_results.tex)) | `draw_pipeline()` (L400-431) | Sơ đồ hộp kiến trúc tĩnh (text→backbone→encoder→heads→loss→prediction) | Không cần — là sơ đồ minh họa. Giữ lại, nhưng xác minh khớp với phương pháp cuối cùng. Không có dữ liệu số. |
| F2 | `fig:perphen` ([`05_results.tex:78`](sections/05_results.tex)) | `draw_per_phenomenon()` (L436-461) | Biểu đồ cột nhóm: macro-F1 theo hiện tượng cho 5 hệ thống (phobert, xlmr, vistral, gpt4o_fs, vipragsent), ±95% CI | Đọc `main_results` (→ [`results/main_pragmatic.json`](results/main_pragmatic.json)). Cùng nguồn với T2. |
| F3 | `fig:mtl` ([`05_results.tex:157`](sections/05_results.tex)) | `draw_mtl_vs_st()` (L466-496) | Cột ghép cặp + mũi tên: PhoBERT đơn nhiệm vụ vs ViPragSent theo hiện tượng | Đọc `main_results` (hàng `phobert_st` và `vipragsent` của [`results/main_pragmatic.json`](results/main_pragmatic.json)). Lưu ý: claim_ledger ánh xạ `fig:mtl` tới `multitask_ablation.json` nhưng code thực sự lấy từ `main_pragmatic` — **sửa lỗi ánh xạ ledger này**. |
| F4 | `fig:sarcasm_lr` ([`05_results.tex:179`](sections/05_results.tex)) | `draw_low_resource()` (L501-523) | Sarcasm F1 vs #ví dụ sarcasm được gán nhãn (trục x log2: 64,128,256,512,1024,2048) cho 5 hệ thống | Ablation tài nguyên thấp thực: huấn luyện lại trên tập con mẫu dương sarcasm. Tạo ra [`results/low_resource_sarcasm.json`](results/low_resource_sarcasm.json). |
| F5 | `fig:confusion` ([`05_results.tex:204`](sections/05_results.tex)) | `draw_confusion()` (L528-551) | Ma trận nhầm lẫn 6×6 theo hàng trên phân cực dụng học (positive, negative, neutral, ironic-pos, sarcastic-neg, mocking) | Số lượng nhầm lẫn thực từ dự đoán ViPragSent trên tập test. Tạo ra [`results/error_confusion.json`](results/error_confusion.json). |
| F6 | `fig:expcurve` ([`05_results.tex:213`](sections/05_results.tex)) | `draw_explanation_curves()` (L556-577) | Pragmatic F1 trên tập test vs epoch (1-10) cho 3 chế độ huấn luyện (direct, CoT-only, full) | Nhật ký đánh giá theo epoch thực (mảng hardcode `base/cot/full` trong code). Cần tệp `results/learning_curves.json` mới (hiện tại CHƯA có trong claim_ledger / chưa có tệp kết quả; ledger sai khi trỏ `fig:expcurve` tới `main_pragmatic.json`). |
| F7 | `fig:calibration` ([`05_results.tex:223`](sections/05_results.tex)) | `draw_calibration()` (L582-604), helper `make_reliability()` (L257-276) | Biểu đồ độ tin cậy (3 bảng: PhoBERT, Vistral, ViPragSent) với chú thích ECE | Độ tin cậy/độ chính xác theo bin thực từ xác suất đầu phân cực. Tạo ra [`results/calibration.json`](results/calibration.json). |
| F8 | `fig:qualitative` ([`06_analysis.tex:14`](sections/06_analysis.tex)) | `draw_qualitative()` (L609-642) | Hai thẻ ví dụ với dự đoán của hệ thống (văn bản hardcode trong hàm) | Các ví dụ test đã được phân xét thực + dự đoán mô hình thực. Nguồn [`data/generated/qualitative.jsonl`](data/generated/qualitative.jsonl) (hiện tại là 2 hàng tổng hợp). |

---

## 4. Bản đồ phụ thuộc dữ liệu → sản phẩm

```
THÍ NGHIỆM THỰC (cần chạy)               TỆP KẾT QUẢ               BẢNG / HÌNH VẼ
---------------------------------         ----------------           ----------------
E0  Thu thập dữ liệu + chú thích  ──────► data/manifest/datasets.json ─► T1 tab:datasets
    (8.4k/2.1k/2.1k pragmatic)             data/.../*.jsonl (annotations)  (+ thống kê α chú thích trong
                                                                           prose 04/appendix B)

E1  Huấn luyện 11 hệ thống,        ──────► results/main_pragmatic.json ──► T2 tab:main_pragmatic
    đánh giá trên tập test                                                   F2 fig:perphen
    (PhoBERT-ST/FT, XLM-R, Sailor,                                          F3 fig:mtl (phobert_st vs vipragsent)
     Vistral, GPT-4o-mini zs/8s,
     ViPragSent + 3 biến thể suy luận)

E2  Đánh giá cùng hệ thống trên    ──────► results/ordinary_sentiment.json► T3 tab:ordinary
    3 tập test cảm xúc công khai

E3  Quét ablation một backbone     ──────► results/multitask_ablation.json► T4 tab:ablation
    (8 cấu hình, tập dev)

E4  Quét sarcasm tài nguyên thấp   ──────► results/low_resource_sarcasm.json►F4 fig:sarcasm_lr
    (lấy mẫu con 64..2048 mẫu dương)

E5  Phân tích nhầm lẫn / lỗi       ──────► results/error_confusion.json ──► F5 fig:confusion

E6  Ghi nhật ký đánh giá theo      ──────► results/learning_curves.json* ─► F6 fig:expcurve
    epoch (trong quá trình chạy E1          (*TỆP MỚI — chưa có trong ledger)
    ViPragSent)

E7  Hiệu chỉnh (đầu phân cực)      ──────► results/calibration.json ──────► F7 fig:calibration

E8  Hạch toán tính toán/API        ──────► results/cost_breakdown.json ───► T5 tab:cost
    (ghi lại trong suốt E1-E7)

E9  Chọn lọc ví dụ định tính       ──────► data/generated/qualitative.jsonl►F8 fig:qualitative

F1 fig:pipeline = sơ đồ minh họa, không phụ thuộc dữ liệu.
```

Lưu ý: `results/calibration.json` chỉ là nguồn cho F7; `result_schema.json`
([`results/result_schema.json`](results/result_schema.json)) là mẫu chung
cho mỗi lần chạy (status=pending) và hiện tại KHÔNG được dùng bởi bảng/hình vẽ nào
— hãy áp dụng nó làm định dạng bản ghi chuẩn cho mỗi lần chạy để lưu trữ thô theo
seed.

---

## 5. Các thí nghiệm cần chạy

### Bộ dữ liệu cần thu thập (theo [`configs/datasets.yaml`](configs/datasets.yaml) — hiện tại là `pending_manifest`)

Cấu hình chỉ là stub; cần được điền với các mục cụ thể. Các kho ngữ liệu yêu cầu:

1. **UIT-VSFC** (`vannguyen2018uitvsfc`) — Cảm xúc phản hồi sinh viên 3 chiều.
   Giấy phép chỉ dùng cho nghiên cứu (cần xin truy cập). Phân chia 11426/1583/3166.
2. **UIT-VSMEC** (`ho2020uitvsmec`) — Cảm xúc Facebook 7 chiều. Chỉ dùng nghiên cứu.
   Phân chia 5548/686/693.
3. **AIVIVN-2019** (`aivivn2019`) — Cảm xúc đánh giá thương mại điện tử 3 chiều.
   Phát hành thi công khai. Phân chia 16087/1788/5454.
4. **ViPragSent (mới, cần xây dựng)** — ~12.622 bình luận từ Facebook (46%),
   TikTok (31%), YouTube (23%), tháng 1–11 năm 2025; 3.209 bình luận tái sử dụng
   từ tập test UIT-VSFC/VSMEC + 9.413 bình luận mới. Phân chia cuối sau chú thích
   8412/2104/2106. CC-BY-NC-4.0. Lưu ý source_note.md định phạm vi này ở 10k–15k
   bình luận; bản thảo cam kết 12.622. Ngưỡng thành công thí điểm
   (`notes/source_note.md`): bắt đầu với 2.000 ví dụ, chỉ sarcasm + implicit, thành
   công nếu multi-task transfer thêm ≥4 macro-F1 so với đơn nhiệm vụ.

Tiền xử lý theo [`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex):
Phân đoạn từ VnCoreNLP (`nguyen2017vntokenizer`), chuẩn hóa từ vựng ViSoLex
(`vlexnorm2024`), lọc trước nội dung độc hại ViHSD/ViHOS.

### Quy trình chú thích (E0)
- 2 người chú thích là người bản ngữ tiếng Việt (L1) mỗi bình luận + 1 người phân
  xét (tiến sĩ ngôn ngữ học).
- Nhãn 9 trường: 6 cờ dụng học nhị phân (implicit, sarcasm, irony, idiom,
  code_switch, mocking) + phân cực ý định 3 chiều + cảm xúc UIT-VSMEC-7 + rationale
  tùy chọn 1–2 câu. Bộ tiêu chí trong
  [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex).
- Mục tiêu: Krippendorff's α ≥ 0.71 tổng thể (bài báo khẳng định theo trường
  0.72/0.74/0.66/0.69/0.84/0.61), Cohen's κ ≥ 0.69 trên quyết định sarcasm-flip.
  Chú thích lại các lô có α trước phân xét < 0.55.

### Mô hình / backbone (theo [`configs/models.yaml`](configs/models.yaml) — hiện tại là `TBD`)
Điền: **PhoBERT-base** (`phobert2020`, backbone triển khai mặc định),
**XLM-R-large** (`conneau2020xlmr`), **Vistral-7B** (`vistral2024`, backbone chính,
QLoRA), **Sailor-7B** (`sailor2024`, QLoRA). GPT-4o-mini
(`openai2024gpt4o`) qua API để tạo rationale + baseline có gợi nhắc.

### Baseline cần cài đặt (11 hệ thống, tên thực từ bài báo)
1. PhoBERT (đơn nhiệm vụ) — một đầu cho mỗi hiện tượng.
2. PhoBERT fine-tune (tinh chỉnh) — một đầu multi-label 6 chiều (điểm tham chiếu chính).
3. XLM-R-large.
4. Sailor-7B SFT (QLoRA r=16, α=32).
5. Vistral-7B SFT (QLoRA r=16, α=32).
6. GPT-4o-mini zero-shot.
7. GPT-4o-mini 8-shot (demo in-context cố định từ tập train).
8. ViPragSent − không có auxiliary loss.
9. ViPragSent − CoT-only (đọc nhãn từ rationale).
10. ViPragSent − explanation-only.
11. **ViPragSent (đầy đủ)** — đầu multi-task + trọng số bất định + đầu rationale
    bỏ qua khi suy luận.

### Quy trình huấn luyện (theo [`configs/training.yaml`](configs/training.yaml) — hiện tại là `seeds:[1,2,3]`, CẦN SỬA thành 5)
- PhoBERT/XLM-R: AdamW, lr 2e-5, batch 32, 10 epoch, dừng sớm theo dev
  macro-pragmatic F1, độ dài chuỗi 128, fp16, grad clip 1.0.
- Vistral/Sailor: QLoRA r=16 α=32 dropout 0.05 trên q,k,v,o; lr 1e-4; batch 16;
  3 epoch; lượng tử hóa NF4.
- Bộ giải mã rationale: 2 lớp, 128-d, 4 đầu, dropout 0.1, teacher forcing,
  trọng số mất mát β=0.3. Log-variance bất định khởi tạo 0, học chung.
- GPT-4o-mini: giải mã tham lam (greedy decoding) với prompt rationale/phân loại
  (Appendix B).

### Đánh giá (theo [`configs/eval.yaml`](configs/eval.yaml))
- Chính: binary macro-F1 theo hiện tượng + macro-pragmatic F1.
- Duy trì cảm xúc thông thường: macro-F1 trên UIT-VSFC, UIT-VSMEC, AIVIVN.
- Hiệu chỉnh (calibration): ECE, 10 bin chiều rộng bằng nhau, trên đầu phân cực
  dụng học; báo cáo Brier.
- **5 seed** (khởi tạo ngẫu nhiên, trộn dữ liệu, dropout) — phải đồng bộ training.yaml
  (hiện tại là 3) với bài báo (5).
- Ý nghĩa thống kê: paired bootstrap (`koehn2004bootstrap`), 1000 lần lấy mẫu lại;
  CI 95%. Kiểm định ý nghĩa kèm p-value hoặc CI-of-difference.
- IAA: Krippendorff's α, Cohen's κ.

---

## 6. Quy trình thay thế (từng bước)

### (a) Chạy thí nghiệm và xuất chỉ số vào `results/*.json`
Chạy E0–E9 (Mục 4). Viết một bộ khung huấn luyện/đánh giá thực (mới
`scripts/train.py`, `scripts/eval.py` — chưa tồn tại) theo "Artifact contract" trong
[`sections/03_method.tex:99-106`](sections/03_method.tex), ghi lại theo từng lần
chạy: ID đầu vào + nền tảng, nhãn gold + dự đoán, logit theo đầu, rationale, seed,
ID backbone, hash script. Xuất bản ghi thô theo seed theo
[`results/result_schema.json`](results/result_schema.json) (điền `metrics.primary`,
`confidence_interval`, `cost`, `seed`, `status:"complete"`), sau đó tổng hợp thành
năm tệp JSON kết quả hiện có **giữ nguyên schema hiện tại** để hình vẽ tiếp tục
hoạt động:
- `main_pragmatic.json`: `{metric, seeds, budget, systems:{sys:{phen:{mean,ci95,seeds[]}, macro_prag}}}`
- `ordinary_sentiment.json`: `{metric, datasets, systems:{sys:{ds:{mean,ci95}}}}`
- `multitask_ablation.json`: `[{name, macro_prag_f1, ord_sent_f1, ece, rel_cost}]`
- `low_resource_sarcasm.json`: `{sys:[{n,mean,ci95}]}`
- `error_confusion.json`: `{labels, counts[6][6]}`
- `calibration.json`: `{sys:{ece, brier, bins:[{low,high,n,conf,acc}]}}`
- `cost_breakdown.json`: `{sys:{compute_gpu_h, compute_usd, api_usd, annotation_usd, total_usd}}`
- **MỚI** `learning_curves.json` cho F6: `{regime:[f1_per_epoch]}` (thêm vào ledger).

### (b) Tái cấu trúc `make_artifacts.py` để ĐỌC kết quả thay vì mô phỏng
Trong [`scripts/make_artifacts.py`](scripts/make_artifacts.py), **xóa bộ mô phỏng
và thay bằng các hàm đọc tệp**. Các hằng số/hàm cụ thể cần xóa hoặc thay thế:
- Xóa `BASE_PHEN` (L97-104), `SYS_DELTA` (L107-130), `ORD_BASE` (L163),
  `ORD_DELTA` (L165-177), `sample_score()` (L132-136), và các vòng lặp tại
  L139-158 và L179-196 → thay bằng `json.load(open(RES/"main_pragmatic.json"))`
  và `.../ordinary_sentiment.json`.
- Xóa danh sách `ABLATIONS` (L201-210) → load `multitask_ablation.json`.
- Xóa vòng lặp đường cong logistic L219-235 → load `low_resource_sarcasm.json`.
- Xóa `CONFUSION` hardcode (L242-249) → load `error_confusion.json`.
- Xóa `make_reliability()` (L257-276) và `CALIB` ECE/Brier hardcode (L278-289)
  → load `calibration.json`.
- Xóa `COSTS` hardcode (L296-321) → load `cost_breakdown.json`.
- Xóa dict `manifest` (L328-348) → `data/manifest/datasets.json` thực trở thành
  nguồn sự thật (script KHÔNG được ghi đè bằng dữ liệu tổng hợp).
- Xóa `QUAL` (L353-370) → load `data/generated/qualitative.jsonl` thực.
- Các mảng per-epoch hardcode `base/cot/full` trong `draw_explanation_curves()`
  (L559-561) → load `learning_curves.json` mới.
- Giữ lại các hàm vẽ `draw_*` (L400-642) — chúng có thể tái sử dụng khi được
  cung cấp mảng thực. Giữ lại khối kiểu hình vẽ (L33-58). Seed RNG (L27) trở nên
  không cần thiết khi bỏ mô phỏng; xóa nó.

### (c) Tạo lại hình vẽ
Sau (b), chạy `python scripts/make_artifacts.py` để kết xuất lại
[`figures/fig2..fig8.pdf`](figures/) từ dữ liệu thực. F1 (`fig1_pipeline.pdf`) là
sơ đồ minh họa — xác minh lại khớp với kiến trúc cuối cùng. Xác nhận F3 đọc từ
cùng nguồn như ledger ghi (sửa ledger hoặc code — xem §7).

### (d) Thay thế các số nội tuyến hardcode trong mỗi bảng
Năm bảng được gõ thủ công trong LaTeX và SẼ bị lệch so với JSON. Cập nhật từng
bảng, sau đó **chuyển sang tự động tạo**:
- **T1 `tab:datasets`** ([`04_experimental_setup.tex:26-29`](sections/04_experimental_setup.tex)):
  số lượng phân chia thực + giấy phép đã xác minh.
- **T2 `tab:main_pragmatic`** ([`05_results.tex:35-46`](sections/05_results.tex)):
  mỗi `\tiny{$\pm$0.5}` là CI giữ chỗ tổng hợp — thay tất cả 11 hàng bằng giá trị
  trung bình thực + CI thực theo ô.
- **T3 `tab:ordinary`** ([`05_results.tex:88-96`](sections/05_results.tex)):
  8 hàng gồm 3 số.
- **T4 `tab:ablation`** ([`05_results.tex:125-132`](sections/05_results.tex)):
  8 hàng; chú ý chú thích `\dag` ECE-scaled-by-10³.
- **T5 `tab:cost`** ([`05_results.tex:247-254`](sections/05_results.tex)):
  8 hàng gồm GPU-h/$/API/tổng.
- **KHUYẾN NGHỊ (ngăn lệch lạc):** thêm `scripts/make_tables.py` để xuất các
  đoạn `.tex` (ví dụ `tables/main_pragmatic.tex`) từ JSON, và dùng `\input{}` trong
  các section thay vì số nội tuyến. Điều này đảm bảo paper == results bằng cấu trúc
  và là bản sửa lỗi nghiêm ngặt có đòn bẩy cao nhất.

### (e) Cập nhật `claim_ledger.csv` và tái xác minh mọi khẳng định số trong văn xuôi
- Sửa các lỗi ánh xạ trong [`results/claim_ledger.csv`](results/claim_ledger.csv):
  `fig:mtl` được ánh xạ tới `multitask_ablation.json` nhưng code đọc
  `main_pragmatic.json`; `fig:expcurve` được ánh xạ tới `main_pragmatic.json` nhưng
  chưa có nguồn thực (thêm `learning_curves.json`). Thêm một hàng cho mỗi khẳng định
  trong văn xuôi.
- Tái xác minh MỌI con số nội tuyến trong văn xuôi so với JSON, bao gồm:
  tóm tắt (`main.tex:44-57`: +5.5, +7.4, +5.9, +2.4/+0.6/+0.5, +7.2/+3.2,
  −1.5/−6.4, ECE 0.094→0.048); các phát hiện trong phần giới thiệu
  ([`01_introduction.tex:70-89`](sections/01_introduction.tex):
  91.5/46.7/58.5, các delta theo hiện tượng, −2.7); văn xuôi kết quả
  ([`05_results.tex`](sections/05_results.tex): +3.3 so với 8-shot, hồi quy GPT
  −3.4/−1.2, 35.6 vs 48.2 @64, khoảng cách +12.6→+7.2, tỷ lệ cháy 0.93,
  nhầm lẫn 2–5% vs 11–18%); phân tích ([`06_analysis.tex`](sections/06_analysis.tex):
  tỷ lệ bỏ sót 61% vs 19%, biên 74%, phân tách +2.7/+3.0/+1.5, hoán đổi backbone
  61.5→59.9, độ trễ 5.1ms/34.7ms); số IAA trong
  [`appendix/b_annotation_and_prompts.tex:57-64`](appendix/b_annotation_and_prompts.tex)
  và tổng tính toán trong [`appendix/a_reproducibility.tex:36-41`](appendix/a_reproducibility.tex)
  (~470 A100-h, $11.3 API).

### (f) Biên dịch lại
`latexmk -pdf main.tex` (hoặc build hiện có của dự án). Xác nhận tất cả
`\ref{tab:*}`/`\ref{fig:*}` được giải quyết và không còn marker tổng hợp nào.

---

## 7. Kiểm tra tính nhất quán & độ nghiêm ngặt

1. **paper == results == figures**: sau khi tự động tạo bảng (§6d) và hình vẽ
   (§6c), so sánh từng ô bảng và giá trị hình vẽ với JSON tương ứng. Khẳng định
   macro-pragmatic "+5.5" trong tóm tắt phải bằng `vipragsent.macro_prag −
   phobert.macro_prag` trong `main_pragmatic.json`.
2. **Các mâu thuẫn nội bộ cần giải quyết NGAY** (có ngay cả trong bản build tổng
   hợp): tóm tắt/giới thiệu nói backbone chính của ViPragSent là **Vistral** (hàng
   T2 "ViPragSent (ours, Vistral)") nhưng phần giới thiệu liệt kê PhoBERT là
   backbone được đặt tên đầu tiên — hãy làm rõ backbone-of-record ở mọi nơi.
   Macro-prag tuyệt đối của T2 (61.5) ≠ T4 (66.1) theo thiết kế (backbone khác +
   dev vs test) — giữ lại tuyên bố từ chối trong chú thích nhưng xác minh điều đó
   đúng khi có kết quả thực. `configs/training.yaml` nói 3 seed, bài báo nói 5 —
   sửa config thành 5.
3. **CI / seed / ý nghĩa thống kê**: mỗi ô T2 hiện hiển thị `±0.5` giống nhau —
   thay bằng CI bootstrap 95% theo ô thực trên 5 seed. Chạy paired bootstrap
   (1000 lần lấy mẫu lại) cho các khoảng cách chính và báo cáo p-value hoặc
   CI-of-difference, không chỉ delta điểm.
4. **Giấy phép bộ dữ liệu / checksum / manifest**: [`data/manifest/datasets.json`](data/manifest/datasets.json)
   KHÔNG có checksum và số lượng tổng hợp; README của nó hứa hẹn "source URL,
   license, checksum, count." Thêm SHA256 thực cho mỗi tệp, URL nguồn thực, và
   xác minh các điều khoản chỉ nghiên cứu của UIT-VSFC/UIT-VSMEC cho phép tái chú
   thích + phát hành nhãn dẫn xuất (phần hạn chế đã hạn chế chỉ phát hành chú thích
   mới cho 3.209 bình luận tái sử dụng — tuân thủ điều này).
5. **Xóa mọi marker tổng hợp**: grep cây thư mục để tìm
   `Synthetic example result - replace before submission` (trong
   [`notes/visual_plan.md`](notes/visual_plan.md)) và docstring bộ mô phỏng
   "Numbers come from a fixed-seed reproducible simulator"
   ([`scripts/make_artifacts.py:1-5`](scripts/make_artifacts.py)). Cập nhật
   [`notes/completion_note.md`](notes/completion_note.md) khi các blocker đã được
   giải quyết.
6. **Đạo đức/IRB**: xác nhận sự đồng ý scraping + điều khoản gỡ xuống + đánh giá
   IRB được mô tả trong [`sections/08_ethics.tex`](sections/08_ethics.tex) và
   [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex)
   đã thực sự diễn ra trước khi phát hành.

---

## 8. Ước tính tài nguyên

Từ ghi chú khả thi của đề xuất ([`notes/source_note.md`](notes/source_note.md):
"Trung bình: 80–140 GPU-giờ, 24–48 GB VRAM, cộng chú thích") và ngân sách bản thảo
(tổng hợp) ([`appendix/a_reproducibility.tex`](appendix/a_reproducibility.tex):
~470 A100-h):

- **Tính toán**: Đề xuất nói 80–140 GPU-h; bản thảo khẳng định ~470 A100-h trên
  tất cả hệ thống × 5 seed (95h PhoBERT/XLM-R + 240h Vistral/Sailor SFT + 80h
  ablation + 55h suy luận). Phạm vi thực tế: **~250–470 A100-giờ** tùy thuộc vào
  việc các quét SFT 7B có chạy đủ 5 seed hay không. VRAM 24–48 GB (QLoRA NF4 vừa
  7B trên một card 48GB). **Trung bình** tổng thể, chủ yếu do hai mô hình QLoRA 7B.
- **API**: GPT-4o-mini cho ~8.4k lần tạo rationale + baseline zero/8-shot trên
  tập test 2.106 ≈ **$11–15 USD** (bản thảo: $11.3).
- **Chú thích (ràng buộc chính)**: ~12.6k bình luận × 2 người chú thích + 1
  người phân xét ở $15/h, 14 tuần ≈ **$1.860 USD** mỗi lần thực hiện toàn bộ kho
  ngữ liệu (con số bản thảo). Thông lượng người phân xét ~1.200 bình luận/tuần.
  Đây là điểm nghẽn thực sự và hạng mục có thời gian chờ lâu nhất — bắt đầu E0
  trước.

---

## 9. Rủi ro & biện pháp giảm thiểu (sắp xếp theo mức độ nghiêm trọng)

1. **Tất cả kết quả đều là giả tạo (blocker).** Không có gì có thể tái tạo; nộp bài
   trong trạng thái này là gian lận nghiên cứu. *Biện pháp giảm thiểu*: thực hiện
   §6 từ đầu đến cuối trước khi đưa ra bất kỳ khẳng định nào; không trích dẫn bất
   kỳ con số hiện tại nào ra bên ngoài.
2. **Chi phí chú thích + IAA trên nhãn chủ quan.** Mục tiêu α cho mocking chỉ là
   0.61; sarcasm chiếm 9% / mocking 4% dữ liệu — các lớp hiếm gặp điều khiển các
   khẳng định chính. *Biện pháp giảm thiểu*: chạy thí điểm 2.000 ví dụ
   sarcasm+implicit trước (ngưỡng thành công source_note ≥+4 F1); ngân sách thời
   gian phân xét; báo cáo α trung thực; nếu α của mocking vẫn thấp, hạ nó xuống
   kết quả phụ.
3. **Mức tăng chính có thể không tái tạo được.** Bộ mô phỏng bao gồm +5.5/+7.4;
   multi-task transfer thực trên các lớp hiếm gặp có thể nhỏ hơn hoặc ồn hơn.
   *Biện pháp giảm thiểu*: đăng ký trước tiêu chí thành công thí điểm; sử dụng ý
   nghĩa thống kê paired-bootstrap, không chỉ delta điểm; sẵn sàng tái cấu trúc nếu
   mức tăng <4 F1.
4. **Giấy phép bộ dữ liệu / tính hợp pháp khi phát hành.** UIT-VSFC/UIT-VSMEC chỉ
   dành cho nghiên cứu; văn bản FB/TikTok/YouTube được scrape đặt ra lo ngại về ToS
   + PII. *Biện pháp giảm thiểu*: chỉ phát hành chú thích dẫn xuất cho các bình
   luận tái sử dụng; làm sạch PII + điều khoản gỡ xuống theo phần đạo đức; phê
   duyệt pháp lý/IRB trước khi phát hành.
5. **Độ trung thực rationale GPT-4o-mini + khả năng tái tạo.** Rationale do mô hình
   tạo ra; mô hình và prompt có thể thay đổi. *Biện pháp giảm thiểu*: ghim phiên
   bản mô hình + tham số giải mã + hash prompt (đã có trong artifact contract); giữ
   kiểm tra trung thực 5%; khẳng định ~8.7% unfaithful-drop phải là con số đo
   thực.
6. **Lệch schema / ledger.** Bảng được gõ thủ công; claim_ledger có ít nhất hai
   ánh xạ sai (fig:mtl, fig:expcurve). *Biện pháp giảm thiểu*: tự động tạo bảng
   (§6d) và thêm kiểm tra CI rằng mọi `\label` xuất hiện trong ledger với tệp theo
   dõi hợp lệ.

---

## 10. Danh sách kiểm tra thực thi

- [ ] Điền `configs/datasets.yaml`, `configs/models.yaml`, `configs/training.yaml`
      (đặt seeds=5), `configs/eval.yaml`, `configs/artifacts.yaml` với giá trị thực.
- [ ] E0: thu thập + làm sạch PII + chú thích kho ngữ liệu ViPragSent
      (8412/2104/2106); tính toán và ghi lại Krippendorff α / Cohen κ thực; đảm bảo
      IRB + giấy phép.
- [ ] Xây dựng `scripts/train.py` + `scripts/eval.py` thực, tuân theo artifact
      contract (nhật ký theo lần chạy, hash script, 5 seed, CI bootstrap).
- [ ] E1: huấn luyện + đánh giá tất cả 11 hệ thống → `results/main_pragmatic.json`.
- [ ] E2: đánh giá cảm xúc thông thường → `results/ordinary_sentiment.json`.
- [ ] E3: quét ablation (8 cấu hình, dev) → `results/multitask_ablation.json`.
- [ ] E4: quét sarcasm tài nguyên thấp → `results/low_resource_sarcasm.json`.
- [ ] E5: phân tích nhầm lẫn → `results/error_confusion.json`.
- [ ] E6: ghi nhật ký theo epoch → `results/learning_curves.json` MỚI.
- [ ] E7: hiệu chỉnh (ECE/Brier/bins) → `results/calibration.json`.
- [ ] E8: hạch toán tính toán/API → `results/cost_breakdown.json`.
- [ ] E9: chọn ví dụ định tính thực → `data/generated/qualitative.jsonl`.
- [ ] Điền `data/manifest/datasets.json` với số lượng thực, URL, giấy phép, SHA256.
- [ ] Tái cấu trúc `scripts/make_artifacts.py` để ĐỌC tất cả JSON (xóa `BASE_PHEN`,
      `SYS_DELTA`, `ORD_*`, `sample_score`, `ABLATIONS`, vòng lặp logistic, `CONFUSION`,
      `make_reliability`, `CALIB`, `COSTS`, `manifest`, `QUAL`, seed RNG).
- [ ] Tạo lại `figures/fig2..fig8.pdf`; tái xác minh sơ đồ fig1.
- [ ] Thêm `scripts/make_tables.py`; dùng `\input` cho các `.tex` đã tạo đối với T1–T5.
- [ ] Cập nhật tất cả 5 bảng và tái xác minh mọi số trong văn xuôi của main.tex +
      tất cả section + cả hai appendix so với JSON.
- [ ] Sửa và hoàn thiện `results/claim_ledger.csv` (sửa fig:mtl, fig:expcurve;
      một hàng cho mỗi khẳng định số).
- [ ] Xóa mọi marker "Synthetic ... replace before submission" và docstring bộ mô
      phỏng; cập nhật `notes/completion_note.md`.
- [ ] Chạy cổng nhất quán (paper == results == figures == ledger); biên dịch lại
      với `latexmk -pdf main.tex`; xác nhận không còn tham chiếu bị treo.
