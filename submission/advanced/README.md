# Thử thách nâng cao — OpenStax Statistics

## Nguồn và phạm vi

Benchmark dùng *Introductory Statistics 2e* của OpenStax/Rice University,
847 trang, giấy phép CC BY-NC-SA 4.0. Giữ attribution và ShareAlike cho các
phần dẫn xuất. Không phân phối PDF hoặc crop ảnh trong repo; tải bản gốc từ
`source.json` và xác minh SHA-256 trước khi index.

Index bao gồm toàn bộ 847 trang: 845 trang có text trích xuất, 524 bảng có cấu
trúc (3.190 hàng), 345 caption hình, 517 instance ảnh và 3.100 chunks. Trang 1–2
là bìa không có text. Hình gốc được tham chiếu bằng trang, caption và bounding
box; mô tả trực quan được viết và kiểm tra thủ công cho các hình có trong
`visual_annotations.json`. Không trích hoặc phân phối ảnh riêng.

Phương pháp không chạy OCR hay vision model trên toàn bộ PDF. Text trang và bảng
được trích bằng PyMuPDF; nội dung ảnh/biểu đồ/sơ đồ được tìm qua caption và các
mô tả đã kiểm chứng. Đây là giới hạn quan trọng: hình không có mô tả thủ công
có thể không được hiểu đầy đủ.

## Tạo index

Từ thư mục gốc repo, dùng Python 3.10+:

```powershell
python -m pip install -r submission\advanced\requirements.txt
python submission\advanced\download_source.py --output "$env:TEMP\OpenStax_Intro_Statistics_2e.pdf"
python submission\advanced\build_index.py --pdf "$env:TEMP\OpenStax_Intro_Statistics_2e.pdf" --output submission\advanced\data --chunk-chars 1600 --overlap-chars 160 --annotations submission\advanced\visual_annotations.json
```

`download_source.py` dừng nếu checksum khác với phiên bản đã benchmark. Không
thay checksum trước khi kiểm tra phiên bản và license của nhà xuất bản.

## Benchmark và chạy agent

`questions.json` chứa 18 câu dev và 30 câu test; gold evidence không được gửi vào
agent. Kiểm tra schema, category minimums và evidence trước khi chạy:

```powershell
python submission\advanced\validate_benchmark.py --questions submission\advanced\questions.json --chunks submission\advanced\data\chunks.json
```

Manifest giữ checksum riêng cho từng split. Tạo manifest sau khi tune trên dev
và trước khi chạy test:

```powershell
python submission\advanced\freeze_benchmark.py
```

Cấu hình được chọn trên dev: TF-IDF, top-k 5, tối đa 5 lượt search, context
12.000 ký tự, cộng điểm cho Table/Figure được gọi tên chính xác. Chạy trên dev
hoặc test:

```powershell
python submission\advanced\run_agent.py --questions submission\advanced\questions.json --chunks submission\advanced\data\chunks.json --split dev --output submission\advanced\results\dev-final --top-k 5 --max-searches 5 --budget 12000
python submission\advanced\run_agent.py --questions submission\advanced\questions.json --chunks submission\advanced\data\chunks.json --split test --output submission\advanced\results\test --top-k 5 --max-searches 5 --budget 12000
```

Đặt `OPENAI_API_KEY` và `OPENAI_MODEL` trong `.env` cục bộ hoặc environment.
Không commit `.env`. Chi phí trong `run.json` là ước tính dựa trên token usage
và mức giá cấu hình lúc chạy, không thay thế hóa đơn provider.

## Kết quả held-out test và chấm thủ công

Tạo review template cho test, đối chiếu từng câu với PDF gốc, điền mọi trường
review, rồi tính metric:

```powershell
python submission\advanced\create_review_template.py --questions submission\advanced\questions.json --split test --output submission\advanced\results\test\manual_reviews.json
python submission\advanced\grade_results.py --questions submission\advanced\questions.json --traces submission\advanced\results\test\traces.json --reviews submission\advanced\results\test\manual_reviews.json --output submission\advanced\results\test\graded_summary.json --split test
```

Mỗi câu được chấm đạt/rớt; abstention sai, citation không hỗ trợ đủ câu trả lời,
lỗi hệ thống và timeout đều không đạt. Kết quả đã chấm:

| Nhóm | Đạt / tổng |
|---|---:|
| Bảng | 6/6 |
| Ảnh, sơ đồ | 3/5 |
| Biểu đồ | 5/5 |
| Multi-hop | 2/4 |
| Text | 3/4 |
| Thiếu evidence | 6/6 |
| **Tổng** | **25/30 (83,33%)** |

Pipeline không đạt ngưỡng bonus 90% (cần ít nhất 27/30); không khai báo đã đạt
bonus. Không có lỗi provider/hệ thống. Median latency là 2.418 giây, tổng
93.538 giây; token usage là 76.851 input + 3.119 output và chi phí ước tính
$0.01082625 theo rates được ghi trong `run.json`.

Các lỗi answerable là `test-image-03`, `test-image-05`, `test-text-04`,
`test-multihop-03` và `test-multihop-04`. Ba câu đầu kết thúc bằng abstention do
thiếu hoặc sai định dạng citation ID; hai câu multi-hop không cite đủ evidence
trên các trang được yêu cầu. Candidate output bị từ chối không được tính là câu
trả lời đạt.

`benchmark_manifest.json` ghi hash test đã đóng băng trước khi chạy. Không dùng
kết quả test để thay đổi cấu hình. Traces đầy đủ, reviews thủ công và metrics
theo nhóm nằm trong `results\test\`.
