# Báo cáo Day 7 — Hoàng Văn Bảo (2A202603549)

## Dữ liệu và benchmark

- **Nguồn:** OpenStax, *Principles of Management*, chương 2 và 13; phạm vi 49 trang PDF (35–58 và 375–399), 165,375 ký tự theo manifest Day 6.
- **Quyền sử dụng:** manifest Day 6 ghi CC BY-NC-SA 4.0. Cần đối chiếu giấy phép ở nguồn trước khi chia sẻ bản trích xuất hoặc gửi nội dung tới dịch vụ ngoài.
- **Vector store/index:** Chroma persistent, collection `chunks_B`, Day 6 index `chroma_v2_b1000`; 224 chunks paragraph-based, giới hạn chunk 1,000 ký tự.
- **Embedding:** `BAAI/bge-small-en-v1.5`, FastEmbed 0.9.1, 384 chiều; truy vấn bằng cùng model với index.
- **Agent model:** `gpt-4o-mini`, OpenAI Chat Completions tool calling. Chạy thật trên 23 câu dev và 7 câu test, không dùng mock; provider errors đều bằng 0.
- **Benchmark:** 21 câu có đáp án + 2 câu thiếu evidence ở dev; 5 câu có đáp án + 2 câu thiếu evidence ở test. Sáu câu test Day 6 đã chuyển sang dev vì kết quả của chúng đã được xem. Test Day 7 gồm câu mới, được giữ lại đến khi chốt cấu hình retrieval.
- **Evidence:** script xác minh nguồn, trang và quote trong chunks; phép đối chiếu chuẩn hóa dấu câu/ký tự hỏng khi trích PDF. Các lý do thiếu evidence đã kiểm tra trên corpus, nhưng cần người nộp rà soát trước khi dùng làm nhãn cuối.

## Retrieval và thử nghiệm

Hit@1 và Evidence Hit@3 tính trên câu có đáp án; Context Evidence Rate kiểm tra evidence trong context cuối. Retrieval latency là median theo câu, không tính thời gian index/build; toàn bộ retrieval local, không phát sinh phí API.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF, top-3, metadata filter, 800 ký tự | So sánh lexical | dev | 13/21 (61.9%) | 17/21 (81.0%) | 5/21 (23.8%) | 1.48 ms median; $0 retrieval |
| Phương pháp 2 | Dense Day 6 B, top-3, filter metadata, 800 ký tự | So sánh semantic với lexical | dev | 13/21 (61.9%) | 17/21 (81.0%) | 7/21 (33.3%) | 11.05 ms median; $0 retrieval |
| Phương pháp 3 | Hybrid RRF (dense + TF-IDF), candidate-10, RRF-k=60, top-3, 800 ký tự | Hợp nhất hai ranking để tăng evidence hit | dev | 13/21 (61.9%) | 20/21 (95.2%) | 6/21 (28.6%) | 12.24 ms median; $0 retrieval |
| Cải thiện 1 | Hybrid RRF, candidate-10, top-5, 800 ký tự | Thêm candidate vào context | dev | 13/21 (61.9%) | 20/21 (95.2%) | 6/21 (28.6%) | 12.46 ms median; $0 retrieval |
| Cải thiện 2 — chọn | Hybrid RRF, candidate-10, top-5, 1,600 ký tự | Tăng context budget để giữ được evidence đã tìm | dev | 13/21 (61.9%) | 20/21 (95.2%) | 15/21 (71.4%) | 11.32 ms median; $0 retrieval |
| Cấu hình đã chọn | Hybrid RRF, candidate-10, top-5, 1,600 ký tự | Chốt theo Evidence Hit@3, context evidence và chi phí local | test | 2/5 (40.0%) | 4/5 (80.0%) | 2/5 (40.0%) | 9.20 ms median; $0 retrieval |
| Baseline | TF-IDF, top-3, 800 ký tự | So sánh held-out | test | 2/5 (40.0%) | 3/5 (60.0%) | 1/5 (20.0%) | 0.52 ms median; $0 retrieval |

Hai lỗi dev đã xem:

1. **q19:** Evidence ở trang 45 không xuất hiện trong top-5 hybrid. Câu hỏi dùng cách diễn đạt “strongly liking or disliking”; đoạn nguồn nói về “those we like and those we dislike” rồi khuyến nghị nhiều perspective/opinions. Ranking cho thấy truy vấn chưa đưa được đoạn đúng vào candidate; cần thử query expansion hoặc rerank trên dev.
2. **q01:** Evidence có ở rank 2 nhưng context cuối chỉ giữ chunk rank 1 ở trang 42. Chunk đầu chiếm 1,110 ký tự gồm citation; chunk chứa evidence không vừa phần budget còn lại. Cách pack hiện tại thêm nguyên chunk theo thứ tự ranking, chưa chọn excerpt liên quan.

Một số lỗi test retrieval đã xem: q23 không có evidence trong top-3 hybrid; q22 và q24 có evidence trong danh sách kết quả nhưng context cuối không giữ được. Chưa chỉnh cấu hình dựa trên test.

- **Trace retrieval:** `starter/results/day7-retrieval/{tfidf-dev,dense-dev,hybrid-dev,hybrid-k5-dev,hybrid-k5-b1600-dev,tfidf-test,hybrid-k5-b1600-test}/`.
- **Lệnh chạy:** xem mục “Benchmark retrieval” trong `starter/README.md`; dùng Python environment Day 6 có Chroma/FastEmbed.
- **Giới hạn:** test retrieval chỉ có 5 câu trả lời được; TF-IDF tokenize đơn giản; context vẫn bỏ sót evidence. Agent đã chạy thật; token usage có ghi nhưng chi phí USD chưa tính.

## Agent và đánh giá

- **Tool/agent:** `agent_runner.py` gọi `OpenAICompatibleController` với tool `search`, dùng dense Chroma Day 6. Mặc định tối đa 3 lượt search, top-3 và budget 1,600 ký tự; lỗi được lưu riêng, không dùng kết quả giả.
- **Model:** `gpt-4o-mini`, cấu hình bằng `.env` local (không commit secret). Chỉ gửi tài liệu tới provider được phép xử lý nội dung này.
- **Cấu hình được đánh giá:** dense Day 6 strategy B, top-3, tối đa 3 lượt search, context budget 1,600 ký tự. Đây là cấu hình của agent runner, **không phải** cấu hình hybrid top-5 đã chọn trong benchmark retrieval; lần chạy này không so sánh chất lượng model giữa hai retrieval method.
- **Chấm thủ công:** Đã đọc toàn bộ 23 câu dev và 7 câu test, gồm 5 câu test có đáp án và 2 câu test thiếu evidence. Citation chỉ được chấm đúng khi có trong context và hỗ trợ phát biểu tương ứng. Nhãn cùng lý do nằm trong `starter/results/agent-{dev,test}/reviews.json`; summary được tạo bằng `agent_metrics.py`.

| Split | Tool use | Grounded answer (0/1/2; trung bình /2) | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm | Provider errors | Token usage |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dev | 23/23 | 7 / 0 / 14; **1.33** (21 câu có đáp án) | 15/23 (65.2%) | 19/19 | 2/2 | 2/21 (9.5%) | 0/23 | 15,039 prompt + 2,766 completion = 17,805 |
| test | 7/7 | 1 / 1 / 3; **1.40** (5 câu có đáp án) | 6/6 | 4/4 | 2/2 | 1/5 (20.0%) | 0/7 | 6,038 prompt + 746 completion = 6,784 |

- **Nhận xét lỗi:** Dev q02 và q13, test q24 bị từ chối nhầm sau khi retrieval không đưa được evidence vào context. Dev q06, q07, q08, q09 và q17 có câu trả lời sai hoặc citation không hỗ trợ câu trả lời. Các trường hợp này được chấm 0; citation ID hợp lệ về cú pháp không tự chứng minh grounding.
- **Số lượt search:** Dev có 20 câu chạy 1 lượt và 3 câu chạy 3 lượt; test có 4 câu chạy 1 lượt và 3 câu chạy 3 lượt.
- **Chi phí:** Runner ghi token usage nhưng không trả giá tiền. Cần tính theo bảng giá của model/provider tại thời điểm chạy; chưa ghi USD để tránh ước lượng sai.
- **Trace agent:** `starter/results/agent-dev/{cases.json,traces.json,reviews.json,summary.json}` và `starter/results/agent-test/{cases.json,traces.json,reviews.json,summary.json}`. Các kết quả nằm trong thư mục bị Git ignore; không commit API key, trace hoặc dữ liệu nguồn.

## Thử thách nâng cao

Chưa thực hiện. Index hiện tại chỉ gồm 49 trang nội dung được chọn, không phải tài liệu từ 100 trang có benchmark bảng, ảnh và biểu đồ.
