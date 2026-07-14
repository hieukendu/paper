# Kế Hoạch Kỹ Thuật: FreshBench-RAG

Project: `EMNLP-2026-freshbench-rag`
Tiêu đề: *FreshBench-RAG: Rolling Time-Stamped Evaluation for Retrieval-Augmented QA*

> Phạm vi tài liệu này: một kế hoạch cụ thể ở cấp độ tệp để chuyển đổi gói bản thảo được tạo tự động này thành một bài báo thực sự, sẵn sàng để nộp. Tài liệu **không** chạy thử nghiệm hay chỉnh sửa bất kỳ tệp nào ngoài việc tạo ra `PLAN.md` này.

---

## 1. Tóm tắt bài báo & trạng thái hiện tại

FreshBench-RAG đề xuất một giao thức đánh giá *rolling, time-stamped* (cuộn theo thời gian có nhãn thời điểm) cho hệ thống hỏi-đáp tăng cường truy hồi (retrieval-augmented QA): xây dựng corpus hàng tháng từ các nguồn tin Reuters/BBC/AP cùng các chỉnh sửa gần đây trên Wikipedia, sinh câu hỏi bằng GPT-4o có kèm xác minh grounding (BERTScore), bộ lọc **contamination detection** (phát hiện nhiễm dữ liệu) ba tầng (trùng lặp 8-gram với C4, gán nhãn parametric-hit (kết quả từ bộ nhớ tham số) hộp đen, kiểm tra ngẫu nhiên 5% bởi con người), cùng các metric theo dõi F1/EM của câu trả lời, precision/recall trích dẫn, **freshness score** (điểm mức độ cập nhật) tổng hợp `F`, hệ số suy giảm theo thời gian `β`, và độ bất ổn định xếp hạng Static-vs-Fresh (Kendall τ). Các kết quả headline tuyên bố rằng Fresh F1 giảm 8–14 điểm so với Static, xếp hạng không đồng nhất trong 38% trường hợp, tắt contamination detection làm Fresh F1 phồng lên ~+4.9 điểm (61% bị ghi nhớ), và `F` tương quan với đánh giá "độ cập nhật câu trả lời" của con người ở ρ=0.81 so với 0.63 của raw F1.

**Toàn bộ kết quả định lượng trong gói này là TỔNG HỢP (SYNTHETIC) và phải được thay thế trước khi nộp.**
Tất cả các con số trong bảng được mã hóa cứng (hardcoded) trực tiếp trong [`sections/*.tex`](sections/) và
[`appendix/*.tex`](appendix/); tất cả năm hình vẽ trong nội dung đều là dữ liệu TikZ/pgfplots được mã hóa cứng (không sinh từ bất kỳ tệp nào); bốn tệp PDF trong [`figures/`](figures/) là đầu ra của trình giả lập với seed cố định
([`scripts/make_artifacts.py`](scripts/make_artifacts.py),
[`scripts/generate_figures.py`](scripts/generate_figures.py)) và **không bao giờ được `\includegraphics`'d vào bài báo**. Các tệp [`results/*.json`](results/) mang một *schema placeholder chung chung*
(`quality/risk/cost`, các phương pháp được đặt tên là "Control"/"Cheap baseline"/"Strong API") **không khớp với bất kỳ bảng nào trong bài báo** — hiện tại không có tệp kết quả nào hỗ trợ bất kỳ con số nào được báo cáo. Hiện **không có thư mục `configs/`** (các dataset/model trong đề xuất chỉ tồn tại dưới dạng văn xuôi). Nhiều mâu thuẫn nội bộ đã tồn tại (xem §7). Hãy xem toàn bộ lớp số liệu là chưa được xây dựng.

---

## 2. Danh mục đầy đủ các BẢNG

Tất cả các bảng đều được viết tay dưới dạng LaTeX nội tuyến với các con số tổng hợp. Hiện tại chưa có bảng nào được sinh từ tệp kết quả. Mục "Phải được tạo ra bởi" liệt kê thử nghiệm thực tế + tệp kết quả sẽ điều khiển bảng đó sau khi tái xây dựng (xem §4–§6).

| # | `\label` | Tệp nguồn:dòng | Các metric / cột được báo cáo | Thử nghiệm thực tế + tệp kết quả phải tạo ra bảng |
|---|----------|------------------|----------------------------|------------------------------------------------------|
| T1 | `tab:dataset-stats` | [`sections/05_results.tex:12`](sections/05_results.tex) | Theo split: Raw / Retained / Tỉ lệ Contam. / IAA(κ); Static 4,082→3,847, Fresh 2,467→2,291, các hàng theo quý, human-verified 510→467 | Xây dựng benchmark + đếm bộ lọc contamination → `results/dataset_stats.json` mới |
| T2 | `tab:main` | [`sections/05_results.tex:41`](sections/05_results.tex) | Static & Fresh **Token F1**, **P_cite** theo retriever (BM25/DPR/ColBERT/SPLADE + GPT-4o), các baseline tham số (GPT-4o-ZS, GPT-4o-CoT), Oracle; ΔF1; ±std trên 3 seed; † mức ý nghĩa thống kê | Đánh giá RAG chính → `results/main_results.json` (tái thiết schema) |
| T3 | `tab:per-condition` | [`sections/05_results.tex:109`](sections/05_results.tex) | Fresh F1 & P_cite theo hệ thống × 4 loại thay đổi (Entity/Number/Relationship/Negation) + số lượng theo danh mục (867/611/524/289); 3 seed; † | Đánh giá phân tầng theo loại thay đổi → `results/per_condition.json` |
| T4 | `tab:time-decay` | [`sections/05_results.tex:177`](sections/05_results.tex) | Fresh F1 tại 1/3/6/12 tháng sau cutoff cho BM25/ColBERT/SPLADE + GPT-4o-ZS; hệ số OLS `β̂`; 3 seed | Phân bins theo thời gian + khớp OLS (Eq. `eq:degradation`) → `results/time_decay.json` |
| T5 | `tab:ablation` | [`sections/06_analysis.tex:12`](sections/06_analysis.tex) | Fresh F1 & Δ so với Full cho w/o TCD/CFT/GV/MRU + các ablation 2 thành phần; ColBERT+GPT-4o; 3 seed | Ablation (loại bỏ thành phần) trên 2,291 mục Fresh → `results/ablations.json` (tái thiết schema) |
| T6 | `tab:error-analysis` | [`sections/06_analysis.tex:45`](sections/06_analysis.tex) | Phân loại lỗi (Truy hồi: độ trễ lập chỉ mục/trôi dạt chủ đề/không khớp từ vựng; Sinh câu: ghi đè thông tin cũ/trích dẫn không có cơ sở/mơ hồ thời gian); Số lượng + Tỉ lệ%; κ=0.74 | Phân loại lỗi thủ công trên 400 trường hợp thất bại toàn hệ thống → `results/error_taxonomy.csv` (tái thiết schema) |
| T7 | `tab:efficiency` | [`sections/06_analysis.tex:79`](sections/06_analysis.tex) | Thời gian xử lý mỗi mục (giây), RAM đỉnh (GB), mục/giờ theo hệ thống + chi phí overhead pipeline | Đo thời gian thực tường (wall-clock profiling) trên 500 mục → `results/efficiency.json` mới |
| T8 | `tab:bertscore-threshold` | [`sections/06_analysis.tex:195`](sections/06_analysis.tex) | Precision/Recall/F1/n_labeled theo θ∈{0.60,…,0.80} cho grounding verification | Quét ngưỡng so với nhãn grounding của con người → `results/bertscore_threshold.json` mới |
| T9 | `tab:leakage` | [`sections/06_analysis.tex:221`](sections/06_analysis.tex) | Số lượng/tỉ lệ rò rỉ (leakage) theo split và F1-drop trên 412 mục đã được con người xác minh | Kiểm toán rò rỉ thủ công + tái chấm điểm đã khử nhiễm → `results/leakage.json` mới |
| T10 | *(không có nhãn)* Ablation mở rộng | [`appendix/d_extended_tables.tex:4`](appendix/d_extended_tables.tex) | Thành phần vs Điểm (Full 58.2 / No protocol control / No uncertainty / No validation filter) | **Schema placeholder; từ vựng nhãn sai** — thay bằng ablation thực hoặc xóa; được hỗ trợ bởi `results/ablations.json` |
| T11 | *(không có nhãn)* Độ bền vững mở rộng | [`appendix/d_extended_tables.tex:20`](appendix/d_extended_tables.tex) | Slice vs Điểm (Seen-style/Hard-language/Noisy/Fresh-shifted) | **Schema placeholder** — thay bằng các slice độ bền vững thực hoặc xóa; được hỗ trợ bởi `results/robustness.json` |
| T12 | `tab:qual-success` | [`appendix/e_qualitative_examples.tex:31`](appendix/e_qualitative_examples.tex) | Các trường hợp thành công định tính (Đầu vào / Đầu ra FreshBench / Đầu ra Baseline) | Các bản ghi thực được tuyển chọn từ replay logs |
| T13 | `tab:qual-comparison` | [`appendix/e_qualitative_examples.tex:61`](appendix/e_qualitative_examples.tex) | So sánh định tính với Hybrid+GPT-4o không lọc | Các bản ghi thực được tuyển chọn từ replay logs |

Lưu ý: `tab:error-analysis` (T6) được **tham chiếu là `tab:error-taxonomy`** trong
[`appendix/e_qualitative_examples.tex:4`](appendix/e_qualitative_examples.tex) — một tham chiếu chéo bị hỏng (xem §7).

---

## 3. Danh mục đầy đủ các HÌNH VẼ

**Lưu ý kiến trúc quan trọng:** năm hình vẽ mà bài báo thực sự hiển thị là **TikZ/pgfplots nội tuyến với dữ liệu được nhúng trực tiếp trong tệp `.tex`** — chúng KHÔNG được tạo ra bởi bất kỳ script nào và KHÔNG đọc bất kỳ tệp nào. Bốn tệp PDF trong [`figures/`](figures/) được sinh bởi hai script là **mồ côi** (không bao giờ được `\includegraphics`'d). [`scripts/generate_figures.py`](scripts/generate_figures.py) thậm chí mô tả một bộ hình vẽ khác (bảng F1/citation decay 2 panel, cột ablation, đường Kendall-τ, biểu đồ hiệu chuẩn) so với những gì bài báo hiển thị. Quá trình tái xây dựng phải chọn MỘT quy trình (khuyến nghị: tệp PDF matplotlib thực đọc từ `results/*.json`, được chèn vào qua `\includegraphics`) và đảm bảo các script, hình vẽ và văn xuôi nhất quán với nhau.

| # | `\label` (hoặc PDF) | Hàm sinh / vị trí | Nội dung hình vẽ | Nguồn dữ liệu thực cần có |
|---|-------------------|-------------------------------|---------------|--------------------------|
| F1 | `fig:main-bar` | TikZ nội tuyến, [`sections/05_results.tex:69–98`](sections/05_results.tex) | Biểu đồ cột Static vs Fresh Token F1, 4 hệ thống | Cùng dữ liệu với T2 → `results/main_results.json` |
| F2 | `fig:per-category` | TikZ nội tuyến, [`sections/05_results.tex:133–166`](sections/05_results.tex) | Fresh F1 theo loại thay đổi, 4 hệ thống × 4 danh mục | Cùng dữ liệu với T3 → `results/per_condition.json` |
| F3 | `fig:time-decay` | TikZ nội tuyến, [`sections/05_results.tex:195–224`](sections/05_results.tex) | Fresh F1 so với số tháng sau cutoff, 3 hệ thống; các đường cong suy giảm | Cùng dữ liệu với T4 → `results/time_decay.json` |
| F4 | `fig:scaling` | TikZ nội tuyến, [`sections/06_analysis.tex:105–132`](sections/06_analysis.tex) | Fresh F1 so với ngân sách truy hồi k∈{1,3,5,10,15}, ColBERT & BM25 | Quét ngân sách truy hồi mới → `results/scaling.json` mới |
| F5 | `fig:cross-domain` | TikZ nội tuyến, [`sections/06_analysis.tex:142–175`](sections/06_analysis.tex) | Fresh F1 theo các domain News/Wikipedia/MedQA-Fresh/LegalBench-Rolling, 4 hệ thống | Đánh giá đa domain (2 domain held-out) → `results/cross_domain.json` mới |
| P1 | `figures/fig1_tradeoff.pdf` *(mồ côi)* | `write_pdf(...fig1...)` [`scripts/make_artifacts.py:79`](scripts/make_artifacts.py); cũng là các panel suy giảm matplotlib [`scripts/generate_figures.py:55–107`](scripts/generate_figures.py) | Cột chất lượng (make_artifacts) / Suy giảm F1+citation (generate_figures) | Ngừng sử dụng hoặc kết nối với dữ liệu suy giảm thực |
| P2 | `figures/fig2_ablation.pdf` *(mồ côi)* | `write_pdf(...fig2...)` [`scripts/make_artifacts.py:85`](scripts/make_artifacts.py); cột matplotlib [`scripts/generate_figures.py:110–156`](scripts/generate_figures.py) | Biểu đồ cột ablation | Kết nối với `results/ablations.json` thực hoặc ngừng sử dụng |
| P3 | `figures/fig3_robustness.pdf` *(mồ côi)* | `write_pdf(...fig3...)` [`scripts/make_artifacts.py:91`](scripts/make_artifacts.py); đường τ matplotlib [`scripts/generate_figures.py:159–195`](scripts/generate_figures.py) | Cột độ bền vững / Kendall-τ theo 12 tháng | Kết nối với dữ liệu rank-instability thực hoặc ngừng sử dụng |
| P4 | `figures/fig4_calibration.pdf` *(mồ côi)* | `write_pdf(...fig4...)` [`scripts/make_artifacts.py:97`](scripts/make_artifacts.py); biểu đồ độ tin cậy matplotlib [`scripts/generate_figures.py:198–244`](scripts/generate_figures.py) | Hiệu chuẩn độ tin cậy trích dẫn | Dữ liệu hiệu chuẩn mới (`results/calibration.json` tái thiết schema) |

---

## 4. Sơ đồ phụ thuộc dữ liệu → artifact

```
EXPERIMENTS (cần chạy)              TỆP KẾT QUẢ (schema cần định nghĩa)    CUNG CẤP CHO BẢNG / HÌNH
─────────────────────────────────────────────────────────────────────────────────────────────────
Xây dựng benchmark + bộ lọc 3 tầng ──▶  results/dataset_stats.json        ──▶  T1 (tab:dataset-stats)
Đánh giá RAG chính (4 retriever × GPT-4o,  results/main_results.json (tái-  ──▶  T2 (tab:main), F1 (fig:main-bar)
  +ZS/+CoT, +Oracle), 3 seed           schema'd theo metric thực)
Đánh giá phân tầng theo loại thay đổi ──▶  results/per_condition.json     ──▶  T3 (tab:per-condition), F2 (fig:per-category)
Phân bins theo thời gian + hệ số OLS  ──▶  results/time_decay.json        ──▶  T4 (tab:time-decay), F3 (fig:time-decay)
Ablation thành phần (ColBERT)         ──▶  results/ablations.json (tái-   ──▶  T5 (tab:ablation), T10, P2
                                            schema'd)
Phân loại lỗi thủ công (400 mục)     ──▶  results/error_taxonomy.csv (tái-──▶  T6 (tab:error-analysis)
                                            schema'd)
Đo thời gian thực tường               ──▶  results/efficiency.json        ──▶  T7 (tab:efficiency)
Quét ngưỡng BERTScore θ               ──▶  results/bertscore_threshold.json──▶  T8 (tab:bertscore-threshold)
Kiểm toán rò rỉ + tái chấm điểm      ──▶  results/leakage.json           ──▶  T9 (tab:leakage)
Quét ngân sách truy hồi k             ──▶  results/scaling.json           ──▶  F4 (fig:scaling)
Đánh giá đa domain (4 domain)         ──▶  results/cross_domain.json      ──▶  F5 (fig:cross-domain), T11
Nghiên cứu Freshness-score vs con người──▶  results/freshness_human.json  ──▶  văn xuôi §5.5/§6.7 (ρ=0.81 v.v.)
Các bucket hiệu chuẩn                 ──▶  results/calibration.json (tái- ──▶  P4 (fig4_calibration)
                                            schema'd)
Provenance từng claim                 ──▶  results/claim_ledger.csv       ──▶  mọi claim số liệu (Prop. 1, appendix C)
```

Thực tế hiện tại: mọi mũi tên trên đây đều **bị đứt** — các bảng đọc literal từ `.tex`, hình vẽ đọc literal từ TikZ, và các tệp JSON chứa một schema placeholder không liên quan. Quá trình tái xây dựng phải kết nối vật lý từng mũi tên.

---

## 5. Các thử nghiệm cần chạy

Hiện **không có thư mục `configs/`**; các dataset/model/giao thức chỉ tồn tại trong
[`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex) và
[`notes/source_note.md`](notes/source_note.md). **Đầu ra đầu tiên: tạo `configs/*.yaml`** khai báo mọi dataset, model, retriever, và ngưỡng dưới đây để việc chạy thử nghiệm có thể tái tạo và kiểm toán được (khuyến nghị: `configs/datasets.yaml`, `configs/models.yaml`, `configs/retrieval.yaml`, `configs/eval.yaml`).

### 5.1 Các dataset cần thu thập / xây dựng
- **Corpus FreshBench chính**: thu thập hàng tháng từ các nguồn RSS feeds của **Reuters, BBC World Service, AP** + các chỉnh sửa gần đây trên **Wikipedia** (diff ≥50 token), cùng các tập con BEIR có metadata về ngày xuất bản. Loại trùng lặp bằng **MinHash LSH** (Jaccard 0.8). Hãy thống nhất cửa sổ thời gian: văn xuôi hiện đang mâu thuẫn (setup ghi **Jan–Dec 2024, 12 tháng**; phần kết quả ghi **Jan 2023–Mar 2025, 26 tháng, 3,847 Static + 2,291 Fresh**) — chọn một và đảm bảo tất cả bảng/hình nhất quán.
- **Split kiểm soát Static** (Wikipedia trước cutoff + BEIR) — cần thiết cho phép so sánh Static-vs-Fresh (T2) và vector tham chiếu rank-instability `r_static` (Eq. `eq:rank-tau`).
- **Tập con được con người xác minh** (~412–467 mục; thống nhất hai con số và mâu thuẫn κ=0.73 vs 0.76).
- **Slice độ bền vững** (~528 mục bị nhiễu đối nghịch) và **slice ablation** (~891 mục).
- **Các domain held-out** cho F5: **MedQA-Fresh** (PubMed) và **LegalBench-Rolling** (ý kiến tòa án Mỹ).
- **Mẫu 5% từ C4 / Common Crawl** ([`raffel2020t5`]) cho trùng lặp contamination 8-gram.
- Ghi lại giấy phép nguồn, URL, checksum, và một tệp `MANIFEST` (theo bất biến timestamp tại appendix C `t_s(i) ≤ t_c(i) ≤ t_e(i)`).

### 5.2 Các model / retriever / baseline (dùng đúng tên trong bài báo)
- **Retriever** (top-k=5): `BM25` (rank-bm25 / PyLucene, k1=1.2 b=0.75), `DPR`
  (`facebook/dpr-question_encoder-multiset-base` + FAISS flat-L2), `ColBERT v2`
  (`colbert-ir/colbertv2.0`, PLAID), `SPLADE` (`naver/splade-cocondenser-ensembledistil`), và
  **Hybrid** (BM25 top-100 + DPR rerank qua RRF k=60). Chỉ mục dày đặc (dense index) dùng `all-MiniLM-L6-v2` +
  `faiss-cpu`. *(Lưu ý: abstract/T2 đánh giá bốn retriever không có Hybrid, nhưng Hybrid được định nghĩa trong setup và sử dụng ở appendix E — quyết định xem Hybrid có nằm trong kết quả chính hay không và đảm bảo tất cả phần nhất quán.)*
- **Generator**: `GPT-4o` (`gpt-4o-2024-08-06`), `Claude 3 Haiku` (`claude-3-haiku-20240307`),
  `Llama-3-8B-Instruct` (llama.cpp, 4-bit GGUF). *(Setup hứa hẹn 5×3 tổ hợp nhưng mọi bảng chỉ báo cáo các cặp GPT-4o + một dòng BM25+Llama trong script bị bỏ mồ côi — hoặc chạy toàn bộ lưới hoặc thu hẹp lại tuyên bố.)*
- **Baseline không truy hồi**: `GPT-4o-ZS`, `GPT-4o-CoT`; **Oracle** (đoạn văn vàng) là giới hạn trên.

### 5.3 Giao thức
- Tháng 1–6 = phát triển (điều chỉnh prompt, chọn ngưỡng: BERTScore τ=0.72/0.70 — thống nhất hai giá trị trong phương pháp vs phân tích; ngưỡng trùng lặp contamination θ=0.4). Tháng 7–12 = test held-out; tất cả kết quả chính từ cửa sổ test.
- Phát hiện contamination ba tầng: gắn cờ trùng lặp 8-gram C4 ≥0.4, gán nhãn parametric-hit hộp đen, kiểm tra ngẫu nhiên 5% bởi con người (3 người gán nhãn, đa số quyết định).
- Sinh câu hỏi qua prompt có cấu trúc của GPT-4o (Appendix B — hiện là stub, phải được hoàn thiện); grounding qua BERTScore F1 ≥ ngưỡng.

### 5.4 Seed, metric, mức ý nghĩa thống kê
- **Seed**: ≥3 (bài báo báo cáo std trên 3 thứ tự lấy mẫu câu hỏi). Khuyến nghị ≥5 để CIs ổn định hơn.
- **Metric**: EM, Token F1, P_cite, R_cite, hệ số suy giảm theo thời gian β (OLS, Eq. `eq:degradation`),
  Knowledge-Cutoff-Gap, freshness score tổng hợp `F = F1 · P_cite · e^(−λ·age)` (λ=0.041), tương quan hạng Kendall τ (Eq. `eq:rank-tau`).
- **Mức ý nghĩa / CIs**: paired bootstrap (n=1,000) cho các dấu † trong T2/T3; bootstrap CIs 95% trên tất cả tổng hợp; kiểm định t một phía trên phần dư OLS với hiệu chỉnh **Benjamini–Hochberg** cho các hệ số; kiểm định nhị thức cho tỉ lệ đảo ngược hạng 38%; kích thước hiệu ứng Cohen's d (tuyên bố 0.62–1.14); kiểm định bootstrap trên hiệu ρ cho tuyên bố freshness-score-vs-con-người.
- **Nghiên cứu con người**: inter-annotator κ cho xác minh (0.73/0.76), phân loại lỗi κ=0.74, xác nhận bộ phân loại loại thay đổi κ=0.71, và nghiên cứu đánh giá "độ cập nhật câu trả lời" với 24 người gán nhãn (ρ=0.81 vs F1 0.63, P_cite 0.71) — tất cả hiện tại là tổng hợp; phải được thu thập thực tế.

### 5.5 Bằng chứng lý thuyết / các claim hình thức (appendix C)
Các bằng chứng trong [`appendix/c_protocol_proofs.tex`](appendix/c_protocol_proofs.tex) mang tính **cấp độ giao thức**,
không phải thực nghiệm, vì vậy chúng không cần thử nghiệm để *đúng* — nhưng chúng đặt ra các nghĩa vụ mà quá trình tái xây dựng phải thỏa mãn:
- **Prop. 1 (tính đầy đủ của ledger)**: mọi claim tổng hợp phải đặt tên một table/figure id, và mọi ô table/figure phải đặt tên một tập replay-tuple. → `results/claim_ledger.csv` phải được mở rộng để ánh xạ *từng* claim số liệu đến tập nguồn của nó (hiện nay nó chỉ liệt kê 12 hàng placeholder chung chung).
- **Prop. 2 (giới hạn chi phí cascade `c_l + q·c_h`)**: chỉ có nghĩa nếu một biến thể cascade/selective-escalation thực sự được chạy; nếu không, hãy xóa hoặc đánh dấu rõ ràng là một phát biểu giao thức chung.
- **Bất biến timestamp `t_s ≤ t_c ≤ t_e`**: phải được thực thi và xác minh theo từng mục trong quá trình xây dựng.

---

## 6. Quy trình thay thế (từng bước)

### (a) Chạy thử nghiệm, xuất ra `results/*.json`
Chạy các thử nghiệm theo §5 và viết một tệp JSON cho mỗi kết quả logic (tên schema trong §4). Định nghĩa một schema thực tế —
schema `results/main_results.json` hiện tại (`rows: [{method, quality, risk, cost}]`) là một placeholder chung chung và phải được thay thế. Schema `main_results.json` được đề xuất:
```json
{"metric_window": "test_m7_12", "seeds": [0,1,2],
 "rows": [{"retriever":"ColBERT","generator":"GPT-4o",
           "static_f1":{"mean":80.1,"std":0.7,"ci95":[...]},
           "fresh_f1":{"mean":71.3,"std":1.2,"ci95":[...]},
           "static_pcite":{...},"fresh_pcite":{...},
           "delta_f1":-8.8,"sig_vs_all":true}]}
```
Lưu trữ các replay tuple theo từng seed và từng mẫu `(x_i, y_is, z_is, c_is)` (appendix C Assumption 1).

### (b) Tái cấu trúc các script để ĐỌC `results/*.json` thay vì giả lập
- [`scripts/make_artifacts.py`](scripts/make_artifacts.py): **xóa dict `DATA` được mã hóa cứng
  (dòng 13)** và toàn bộ đường dẫn trình giả lập seed cố định. Tệp này hiện đang *ghi* JSON/CSV tổng hợp và các PDF mồ côi; hãy tái sử dụng nó (hoặc thay thế) để *tổng hợp các log chạy thực tế theo từng seed* thành các schema trong §4, tính mean/std/CI/mức ý nghĩa, và ghi `results/*.json` + `results/claim_ledger.csv`. Xóa mọi sử dụng literal
  `'Synthetic example result - replace before submission'` (khóa `synthetic_label`).
- [`scripts/generate_figures.py`](scripts/generate_figures.py): **thay thế mọi mảng numpy được mã hóa cứng**
  (`hybrid_f1`, `colbert_f1`, `bm25_llama_f1`, `ablation_scores` dòng 121, `tau_values` dòng 163,
  `observed_*` dòng 207–209, v.v.) bằng `json.load()` của `results/*.json` tương ứng. Đồng thời
  thống nhất bộ hình vẽ: các hình vẽ của script này (các panel suy giảm, đường τ, hiệu chuẩn) **không khớp**
  với các hình vẽ TikZ nội tuyến của bài báo.

### (c) Tái sinh hình vẽ — và quyết định quy trình hình vẽ
Hai hướng khả thi; chọn một và áp dụng nhất quán:
- **Hướng A (ít trôi dạt nhất):** giữ năm hình vẽ TikZ nội tuyến nhưng tự động sinh các khối tọa độ của chúng
  từ `results/*.json` qua một bước tạo mẫu nhỏ, để F1–F5 không bao giờ mâu thuẫn với T2–T5.
- **Hướng B:** chuyển đổi F1–F5 sang `\includegraphics` các tệp PDF matplotlib thực được tạo bởi
  `generate_figures.py`, và xóa/tái sử dụng bốn `figures/*.pdf` mồ côi. Dù theo hướng nào, hãy đảm bảo
  các PDF được gửi là bản thực và bốn PDF placeholder được xóa.

### (d) Thay thế các con số nội tuyến được mã hóa cứng trong mỗi bảng `.tex` (liệt kê lại)
Với mỗi bảng dưới đây, ghi đè các con số literal bằng giá trị được lấy từ tệp kết quả tương ứng.
**Khuyến nghị mạnh: tự động sinh các bảng này** (ví dụ: `pgfplotstable` đọc từ CSV, hoặc một bước build xuất ra phần thân bảng `.tex` từ JSON) để loại bỏ sự trôi dạt đã tồn tại giữa văn xuôi, bảng và các tệp `results/`.
- T1 `tab:dataset-stats` — [`sections/05_results.tex:13–27`](sections/05_results.tex)
- T2 `tab:main` — [`sections/05_results.tex:42–60`](sections/05_results.tex)
- T3 `tab:per-condition` — [`sections/05_results.tex:110–124`](sections/05_results.tex)
- T4 `tab:time-decay` — [`sections/05_results.tex:178–188`](sections/05_results.tex)
- T5 `tab:ablation` — [`sections/06_analysis.tex:13–28`](sections/06_analysis.tex)
- T6 `tab:error-analysis` — [`sections/06_analysis.tex:46–62`](sections/06_analysis.tex)
- T7 `tab:efficiency` — [`sections/06_analysis.tex:80–94`](sections/06_analysis.tex)
- T8 `tab:bertscore-threshold` — [`sections/06_analysis.tex:196–206`](sections/06_analysis.tex)
- T9 `tab:leakage` — [`sections/06_analysis.tex:222–229`](sections/06_analysis.tex)
- T10/T11 bảng mở rộng — [`appendix/d_extended_tables.tex`](appendix/d_extended_tables.tex)
  (thay thế từ vựng placeholder hoặc xóa)
- T12/T13 định tính — [`appendix/e_qualitative_examples.tex`](appendix/e_qualitative_examples.tex)
  (thay thế các bản ghi bịa đặt bằng replay logs thực)
- Dữ liệu hình vẽ TikZ nội tuyến: các khối tọa độ F1–F5 (vị trí trong §3).

Đồng thời sửa mọi **con số trong văn xuôi** có tính toán lại từ giá trị bảng: abstract (3,847/2,291; 71.3/80.1;
−13.4; 53–61; +4.9; 38%; ρ=0.81/0.63), §1 đóng góp (dòng 4, 11, 12), tường thuật §5 (dòng 63–67,
191, 193, 226, 234, 236, 240), tường thuật §6 (dòng 31–35, 65–69, 99, 134–136, 177–179, 183, 209, 232),
kết luận §9 (lưu ý: kết luận ghi **4.8** F1 inflation so với **4.9** của abstract, và "2.3× faster"
citation decay chỉ xuất hiện ở §2/§9 — phải được tạo ra và thống nhất).

### (e) Cập nhật claim_ledger.csv và tái xác minh mọi claim số liệu
[`results/claim_ledger.csv`](results/claim_ledger.csv) hiện có 12 hàng placeholder chung chung
(`Control 47.6`, `Cheap baseline`, v.v.) không liên quan đến bài báo. Tái xây dựng nó thành: `claim_id, value, CI,
table_or_figure_label, replay_tuple_set, status` bao gồm **mọi** claim số liệu trong văn xuôi, bảng,
hình vẽ, và trong các ghi chú phản hồi review ([`reviews/author_response.md`](reviews/author_response.md),
[`reviews/round2_author_response.md`](reviews/round2_author_response.md),
[`notes/revision_execution_report.md`](notes/revision_execution_report.md)) — các tài liệu phản hồi đó trích dẫn
các con số phải khớp với kết quả cuối. Trạng thái phải thay đổi từ nhãn tổng hợp sang một con trỏ provenance thực tế (thỏa mãn Prop. 1 appendix C).

### (f) Tái biên dịch
Tái xây dựng `main.pdf` (latexmk). Sửa các tham chiếu bị hỏng trong §7 trước, sau đó xác nhận không còn tham chiếu `??`, không có trích dẫn không xác định (ví dụ: `golchin2024time`), và không còn dấu `\syntheticlabel{}` / `\todoexp{}` nào.

---

## 7. Kiểm tra tính nhất quán & chặt chẽ

**Các mâu thuẫn hiện có PHẢI được giải quyết (phát hiện trong quá trình phân tích):**
1. **Cửa sổ đánh giá**: setup ([`04_experimental_setup.tex:5`](sections/04_experimental_setup.tex))
   ghi "Jan–Dec 2024, cửa sổ 12 tháng, 1,847 mục chính, 412 được con người xác minh, κ=0.73, 1,731 retained";
   kết quả ([`05_results.tex:6`](sections/05_results.tex)) ghi "26 tháng Jan 2023–Mar 2025, 3,847
   Static + 2,291 Fresh, 467 được con người xác minh, κ=0.76". Hai mô tả này mô tả hai benchmark khác nhau.
2. **Giá trị inflation do contamination**: abstract & §5 ghi **+4.9** F1 (CI 3.9–5.5); kết luận
   ([`09_conclusion.tex:4`](sections/09_conclusion.tex)) ghi **4.8**. Script hình vẽ mồ côi dùng
   **77.1 vs 72.3 → +4.8**; T5 dùng **76.2 vs 71.3 → +4.9**.
3. **Tham chiếu chéo bị hỏng**: `\ref{tab:dataset-matrix}` được dùng trong
   [`03_method.tex:13`](sections/03_method.tex) và [`04_experimental_setup.tex:5`](sections/04_experimental_setup.tex)
   — không có nhãn đó (nhãn thực: `tab:dataset-stats`). `\ref{tab:error-taxonomy}` trong
   [`e_qualitative_examples.tex:4`](appendix/e_qualitative_examples.tex) — nhãn thực là
   `tab:error-analysis`. Sẽ hiển thị là `??`.
4. **Trích dẫn không xác định**: `\citep{golchin2024time}` ([`06_analysis.tex:214`](sections/06_analysis.tex))
   — `references.bib` chỉ định nghĩa `golchin2023time`.
5. **Ngưỡng BERTScore**: phương pháp dùng **τ=0.72** ([`03_method.tex:10`](sections/03_method.tex)) và
   setup ghi 0.72; §6.7 phân tích / T8 dùng **θ=0.70** làm mặc định. Cần thống nhất.
6. **Bộ hệ thống không khớp**: T2 đánh giá 4 retriever (không có Hybrid); setup định nghĩa 5 (bao gồm Hybrid);
   appendix E "baseline tốt nhất" là Hybrid+GPT-4o; fig1 mồ côi vẽ BM25+**Llama-3-8B**. Quyết định bộ hệ thống chuẩn tắc.
7. **Kích thước mẫu không khớp**: §6 phân tích lỗi ghi **400** dự đoán thất bại
   ([`06_analysis.tex:39`](sections/06_analysis.tex)) nhưng appendix E ghi "phân tích lỗi **300** mẫu"
   ([`e_qualitative_examples.tex:36`](appendix/e_qualitative_examples.tex)).
8. **`results/dataset_matrix.csv`** (1,862 / 386 / 513 / 913) mâu thuẫn với cả văn xuôi setup và T1.

**Cổng chặt chẽ tiêu chuẩn (áp dụng sau khi tái xây dựng):**
- paper == results == figures: mọi con số trong `.tex` bằng giá trị trong `results/*.json` tương ứng
  (lý tưởng là được thực thi bằng auto-generation).
- CIs + ≥3 (tốt nhất ≥5) seed trên mọi tổng hợp; các kiểm định mức ý nghĩa thực sự được tính (paired
  bootstrap, kiểm định t BH-corrected, nhị thức, Cohen's d) thay vì khẳng định.
- Giấy phép dataset, checksum, và `MANIFEST` được ghi lại; bất biến timestamp `t_s ≤ t_c ≤ t_e` được xác minh theo từng mục; không có mục Fresh nào rò rỉ vào phần kiểm soát Static.
- Xóa mọi dấu "Synthetic example result - replace before submission": macro `\syntheticlabel`
  ([`main.tex:26`](main.tex)) và các sử dụng của nó trong [`appendix/a_reproducibility.tex`](appendix/a_reproducibility.tex),
  [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex),
  [`appendix/c_protocol_proofs.tex`](appendix/c_protocol_proofs.tex),
  [`appendix/d_extended_tables.tex`](appendix/d_extended_tables.tex); các khóa `synthetic_label` trong tất cả
  `results/*.json`; và cột `status` trong `results/claim_ledger.csv`.
- Hoàn thiện các mẫu prompt/gán nhãn stub trong
  [`appendix/b_annotation_and_prompts.tex`](appendix/b_annotation_and_prompts.tex) (hiện tại là khung chung chung, không phải các prompt FreshBench thực tế).

---

## 8. Ước tính tài nguyên

Đây là một thiết kế cố ý **chỉ dùng CPU + API** ("cheap" — chi phí thấp) — không cần GPU
([`04_experimental_setup.tex:20–21`](sections/04_experimental_setup.tex), được xác nhận bởi
[`notes/source_note.md`](notes/source_note.md): ưu tiên 32 GB RAM, 100–200 GB lưu trữ, ngân sách API/gán nhãn vừa phải, thời gian 16–20 tuần).

- **Tính toán**: một máy trạm CPU 16 nhân đơn, 32–64 GB RAM, ~100–200 GB lưu trữ cho các corpus hàng tháng + chỉ mục FAISS/ColBERT. Một chu kỳ đánh giá hàng tháng ≈ 4.5 giờ thực tường (theo tuyên bố); đánh giá 2,291 mục Fresh ≈ 2.2 giờ.
- **API**: GPT-4o ≈ $0.027/mục; toàn bộ cửa sổ 12 tháng tuyên bố là **$214** (GPT-4o $183 + Claude Haiku $31); rolling 26 tháng ≈ **$1,612**. Cộng thêm các lần gọi sinh câu hỏi + judge + bộ phân loại loại thay đổi + scaffolding nghiên cứu con người về freshness-score — ngân sách **~$300–$2,000** tùy thuộc vào cửa sổ được chọn và việc có chạy toàn bộ lưới model 5×3 không. Llama-3-8B cục bộ qua llama.cpp tốn thời gian CPU, không tốn chi phí API.
- **Nỗ lực gán nhãn**: kiểm tra ngẫu nhiên 5% trên mỗi lát cắt hàng tháng (3 người gán nhãn), xác minh 412–467 mục bởi con người, phân loại lỗi 400 mục (2 người gán nhãn), xác nhận loại thay đổi, và **nghiên cứu đánh giá độ cập nhật câu trả lời với 24 người gán nhãn**. Ở mức $18/giờ (theo tuyên bố đạo đức), ước tính **~$1.5–4k** cho gán nhãn, cộng thêm chi phí tuyển dụng/IRB. Đây là ràng buộc chi phí/rủi ro chủ đạo, không phải tài nguyên tính toán.
- **Tổng thể**: **Thấp–Trung bình** — khả thi trên laptop/máy trạm; các ràng buộc chính là ngân sách API và thông lượng gán nhãn bởi con người, không phải phần cứng.

---

## 9. Rủi ro & biện pháp giảm thiểu (theo thứ tự mức độ nghiêm trọng)

1. **Toàn bộ lớp số liệu là tổng hợp và bị ngắt kết nối.** Các bảng đọc literal từ `.tex`, hình vẽ đọc literal từ TikZ, `results/*.json` chứa schema placeholder không liên quan, và bốn PDF là mồ côi.
   *Biện pháp giảm thiểu:* tái xây dựng theo §6 — kết nối experiments→JSON→tables/figures và auto-generate để tránh tái trôi dạt.
2. **Các tuyên bố nghiên cứu con người bị bịa đặt** (các giá trị κ, ρ=0.81 độ cập nhật câu trả lời, nghiên cứu 24 người gán nhãn, phân loại lỗi κ=0.74). Đây là những đóng góp trung tâm của bài báo và không thể làm giả một cách rẻ tiền.
   *Biện pháp giảm thiểu:* lập ngân sách và chạy gán nhãn thực (§8); nếu không khả thi, làm mềm/xóa tuyên bố freshness-vs-con-người.
3. **Các mâu thuẫn nội bộ (8 mục liệt kê trong §7)** sẽ bị mọi reviewer cẩn thận phát hiện và báo hiệu sự bịa đặt. *Biện pháp giảm thiểu:* giải quyết tất cả trước khi tái biên dịch; thêm một kiểm tra CI đảm bảo các con số trong văn xuôi khớp với `results/`.
4. **Cấp phép / phân phối lại dataset** (Reuters/AP/BBC). Bài báo tuyên bố giấy phép học thuật + không phân phối lại văn bản thô. *Biện pháp giảm thiểu:* xác minh điều khoản của từng nguồn, chỉ phát hành các cặp QA + URL + metadata + checksum; ưu tiên các nguồn Wikipedia CC-BY-SA và domain công cộng nếu có thể.
5. **Tính hợp lệ của contamination detection** phụ thuộc vào mẫu C4 5% xấp xỉ dữ liệu pretraining của model đóng — một xấp xỉ mà chính bài báo đánh dấu. *Biện pháp giảm thiểu:* báo cáo nó như một proxy có phân tích độ nhạy; đừng quá tuyên bố "truy hồi thực" mà không thực sự chạy kiểm tra hộp đen + con người.
6. **Sự chồng lấp giữa generator và evaluator câu hỏi** (GPT-4o vừa sinh vừa được đánh giá). *Biện pháp giảm thiểu:* giữ phần loại trừ parametric-hit + kiểm tra ngẫu nhiên bởi con người; cân nhắc dùng một generator câu hỏi khác GPT-4o cho một tập con không thiên lệch.
7. **Nhiễu độ trễ lập chỉ mục / mơ hồ thời gian** trên các mục rất mới có thể chiếm ưu thế trong các thanh lỗi (error bars).
   *Biện pháp giảm thiểu:* cờ độ trễ lập chỉ mục rõ ràng và cutoff 72 giờ được thảo luận trong appendix E; báo cáo CIs.

---

## 10. Danh sách kiểm tra thực thi

- [ ] Tạo `configs/*.yaml` (datasets, models, retrieval, eval, ngưỡng) — hiện đang thiếu.
- [ ] **Giải quyết 8 mâu thuẫn nội bộ trong §7** và sửa các tham chiếu bị hỏng (`tab:dataset-matrix`,
      `tab:error-taxonomy`) và trích dẫn không xác định `golchin2024time`.
- [ ] Quyết định cửa sổ đánh giá chuẩn tắc (12 tháng vs 26 tháng) và bộ hệ thống (4 vs 5 retriever, chỉ GPT-4o
      vs toàn bộ lưới 5×3).
- [ ] Xây dựng các corpus hàng tháng (Reuters/BBC/AP + Wikipedia + BEIR), loại trùng lặp (MinHash 0.8), ghi
      giấy phép/checksum/MANIFEST, thực thi `t_s ≤ t_c ≤ t_e`.
- [ ] Sinh + xác minh grounding câu hỏi (hoàn thiện prompt Appendix B); chạy bộ lọc contamination 3 tầng.
- [ ] Chạy đánh giá RAG chính (T2/F1), phân tầng theo điều kiện (T3/F2), suy giảm thời gian+OLS (T4/F3), ablation (T5), hiệu suất (T7), quét BERTScore (T8), kiểm toán rò rỉ (T9), quét k (F4), đa domain (F5), hiệu chuẩn (P4) — tất cả với ≥3 (mục tiêu ≥5) seed.
- [ ] Chạy nghiên cứu con người: xác minh (κ), phân loại lỗi (400 mục, κ), xác nhận loại thay đổi (κ),
      đánh giá độ cập nhật câu trả lời với 24 người gán nhãn (ρ).
- [ ] Tính CIs + mức ý nghĩa (paired bootstrap, kiểm định t BH-corrected, nhị thức, Cohen's d).
- [ ] Định nghĩa schema `results/*.json` thực tế; xuất dữ liệu theo từng seed + replay tuple.
- [ ] Tái cấu trúc `scripts/make_artifacts.py` (xóa dict `DATA` / trình giả lập; tổng hợp log thực) và
      `scripts/generate_figures.py` (thay thế tất cả mảng được mã hóa cứng bằng `json.load`).
- [ ] Tái sinh hình vẽ (chọn Hướng A inline-from-JSON hoặc Hướng B includegraphics); xóa các PDF mồ côi.
- [ ] Thay thế mọi con số nội tuyến trong bảng (T1–T13) và mọi con số trong văn xuôi; auto-generate các bảng.
- [ ] Tái xây dựng `results/claim_ledger.csv` với provenance thực tế (thỏa mãn appendix C Prop. 1); thống nhất
      các con số được trích dẫn trong các ghi chú phản hồi `reviews/`.
- [ ] Xóa tất cả các sử dụng `\syntheticlabel{}`, các khóa JSON `synthetic_label`, và cột `status` tổng hợp.
- [ ] Tái biên dịch `main.pdf`; xác minh không còn `??`, không có trích dẫn không xác định, không còn dấu tổng hợp.
- [ ] Kiểm tra nhất quán cuối: paper == results == figures == claim_ledger == phản hồi review.
