# Báo cáo Day 7 — Hoàng Văn Bảo (2A202603549)

## Dữ liệu và benchmark

- **Nguồn:** OpenStax, *Principles of Management*, chương 2 và 13; phạm vi 49 trang PDF (35–58 và 375–399), 165,375 ký tự theo manifest Day 6.
- **Quyền sử dụng:** manifest Day 6 ghi CC BY-NC-SA 4.0. Cần đối chiếu giấy phép ở nguồn trước khi chia sẻ bản trích xuất hoặc gửi nội dung tới dịch vụ ngoài.
- **Vector store/index:** Chroma persistent, collection `chunks_B`, Day 6 index `chroma_v2_b1000`; 224 chunks paragraph-based, giới hạn chunk 1,000 ký tự.
- **Embedding:** `BAAI/bge-small-en-v1.5`, FastEmbed 0.9.1, 384 chiều; truy vấn bằng cùng model với index.
- **Agent model:** Chưa chạy agent thật vì môi trường hiện tại chưa cấu hình `OPENAI_API_KEY` và `OPENAI_MODEL`. Đã thêm controller tương thích OpenAI Chat Completions; không dùng mock để báo cáo chất lượng.
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
- **Giới hạn:** test chỉ có 5 câu trả lời được; TF-IDF tokenize đơn giản; context vẫn bỏ sót evidence. Phí model trả lời chưa đo vì agent chưa chạy.

## Agent và đánh giá

- **Tool/agent:** `agent_runner.py` gọi `OpenAICompatibleController` với tool `search`, dùng dense Chroma Day 6. Mặc định tối đa 3 lượt search, top-3 và budget 1,600 ký tự; lỗi được lưu riêng, không dùng kết quả giả.
- **Model:** Chưa cấu hình/chạy. Thiết lập `OPENAI_API_KEY`, `OPENAI_MODEL` (và tùy chọn `OPENAI_BASE_URL`) trong môi trường local; không commit secret. Chỉ gửi tài liệu tới provider được phép xử lý nội dung này.
- **Grounded answer, citation support, abstention:** Chưa có kết quả agent thật; cần chạy model, đọc từng trace/citation và chấm thủ công. Các test mock chỉ kiểm tra code, không phải benchmark chất lượng.
- **Trace agent và nhãn:** Chưa tạo. `agent_runner.py` ghi `cases.json` và `traces.json`; tạo `reviews.json` thủ công trước khi chạy `agent_metrics.py`. Token usage được ghi khi provider trả usage; chi phí phải tính theo giá của model/provider.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent thật | Chưa chạy — thiếu cấu hình provider | — | — | — | — | — | — |

## Thử thách nâng cao

Chưa thực hiện. Index hiện tại chỉ gồm 49 trang nội dung được chọn, không phải tài liệu từ 100 trang có benchmark bảng, ảnh và biểu đồ.
