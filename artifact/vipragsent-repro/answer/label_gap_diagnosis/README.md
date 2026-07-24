# Chẩn đoán và tối ưu nhãn irony / idiom / mocking

Đây là artefact kỹ thuật độc lập; không chỉnh sửa manuscript. Chẩn đoán test chỉ dùng để mô tả các khoảng cách đã công bố. Mọi checkpoint, threshold, và lựa chọn source mới đều dùng `vipragsent_train.jsonl` và `vipragsent_dev.jsonl`, không dùng nhãn test để chọn.

## Chẩn đoán nguyên nhân

| Nhãn | Source ViPragSent hiện tại | Lỗi test | Diễn giải |
| --- | --- | --- | --- |
| irony | ViSoBERT weighted encoder, threshold dev 0.59 | 17 FN, 0 FP | Mô hình bỏ sót các irony xác suất thấp; không thể sửa bằng đổi threshold đơn thuần mà không tạo FP. Các FN đồng xuất hiện nhiều với mocking (52.94%) và sarcasm (47.06%). |
| idiom_figurative | PhoBERT weighted encoder, threshold dev 0.66 | 17 FN, 8 FP | FN bằng các PhoBERT baseline seed mạnh nhất, nhưng source hiện tại sinh thêm FP; các FP đồng xuất hiện implicit sentiment/sarcasm. |
| mocking | ViSoBERT weighted encoder, threshold dev 0.65 | 107 FN, 71 FP | Lỗi hai chiều và phụ thuộc ngữ cảnh sarcasm/implicit; expert một-nhãn mất context nên không bền qua ba seed. |

`diagnosis.json` lưu toàn bộ confusion, prevalence, lỗi theo source và co-occurrence; nó không được dùng để fit checkpoint hoặc threshold.

## Can thiệp đã chạy

Hai config cố định được lưu dưới `configs/`:

- `label_gap_expert_triage.yaml`: expert một-nhãn. Irony và idiom không vượt source dev; mocking chỉ thắng ở một seed nhưng ensemble ba seed không giữ được lợi thế.
- `label_gap_joint_targeted_phase2.yaml`: giữ học đa nhãn, tăng weight của nhãn đích lên 3. Irony và idiom không vượt source dev. Mocking joint three-seed ensemble đạt 81.4572 dev F1, vượt 80.5806 của source locked.

Hybrid dev-only cuối thay source mocking bằng ensemble joint targeted; các nhãn khác giữ source locked.

| Metric dev | Hybrid locked | Hybrid nhãn-đích cuối |
| --- | ---: | ---: |
| Implicit | 67.5939 | 67.5939 |
| Sarcasm | 80.8124 | 80.8124 |
| Irony | 97.6816 | 97.6816 |
| Idiom | 97.0483 | 97.0483 |
| Code-switch | 80.3963 | 80.3963 |
| Mocking | 80.5806 | 81.4572 |
| Macro-F1 | 84.0188 | 84.1649 |

## Artefact

- `diagnosis.json`: audit lỗi test hậu nghiệm.
- `dev_predictions/`, `dev_thresholded/`, `thresholds/`: prediction dev và calibration của từng candidate.
- `selection/final_dev_selection.json`, `selection/final_dev_predictions.jsonl`, `selection/final_dev_metrics.csv`: hybrid chọn hoàn toàn trên dev.
- `outputs/label_gap_experts/`: checkpoint, manifest và history gốc cho mọi run, bao gồm các kết quả không vượt ngưỡng.

## Trạng thái kết luận

Không có test evaluation mới trong bundle này. Vì test lịch sử đã được quan sát trước khi chẩn đoán, bundle **không** thiết lập claim rằng ViPragSent đã vượt mọi best-tuned baseline ở irony, idiom, mocking hoặc mọi metric. Một test set mới/chưa quan sát là cần thiết cho claim đó.
