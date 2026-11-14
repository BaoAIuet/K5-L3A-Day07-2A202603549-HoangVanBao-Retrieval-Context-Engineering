# Báo cáo Day 7 — Hoàng Văn Bảo (2A202603549)

## Dữ liệu và benchmark

- **Nguồn:** OpenStax, *Principles of Management*, chương 2 và 13; phạm vi 49 trang PDF (35–58 và 375–399), 165,375 ký tự theo manifest Day 6.
- **Quyền sử dụng:** _Principles of Management_ của OpenStax/Rice University có giấy phép CC BY-NC-SA 4.0 theo [trang license chính thức](https://openstax.org/books/principles-management/pages/preface). Các trace có trích đoạn được chia sẻ phi thương mại kèm attribution trong `submission/results/README.md`.
- **Vector store/index:** Chroma persistent, collection `chunks_B`, Day 6 index `chroma_v2_b1000`; 224 chunks paragraph-based, giới hạn chunk 1,000 ký tự.
- **Embedding:** `BAAI/bge-small-en-v1.5`, FastEmbed 0.9.1, 384 chiều; truy vấn bằng cùng model với index.
- **Agent model:** `gpt-4o-mini`, OpenAI Chat Completions tool calling. Chạy dense baseline và hybrid được chọn, mỗi cấu hình gồm 23 câu dev và 7 câu test; provider errors đều bằng 0.
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

- **Trace retrieval:** `submission/results/retrieval/{tfidf-dev,dense-dev,hybrid-dev,hybrid-k5-dev,hybrid-k5-b1600-dev,tfidf-test,hybrid-k5-b1600-test}/`.
- **Lệnh chạy:** xem mục “Benchmark retrieval” trong `starter/README.md`; dùng Python environment Day 6 có Chroma/FastEmbed.
- **Giới hạn:** test retrieval chỉ có 5 câu trả lời được; TF-IDF tokenize đơn giản; context vẫn bỏ sót evidence. Chi phí agent là ước tính theo token và bảng giá model, không phải số tiền billing xác nhận.

## Agent và đánh giá

- **Tool/agent:** `agent_runner.py` gọi `OpenAICompatibleController` với tool `search`; hỗ trợ dense Chroma Day 6 và hybrid RRF (dense + TF-IDF). Đánh giá baseline dense và cấu hình hybrid đã chọn; mỗi cấu hình tối đa 3 lượt search, budget 1,600 ký tự; lỗi được lưu riêng, không dùng kết quả giả.
- **Model:** `gpt-4o-mini`, cấu hình bằng `.env` local (không commit secret). Chỉ gửi tài liệu tới provider được phép xử lý nội dung này.
- **Cấu hình được đánh giá:** Dense baseline: Day 6 strategy B, top-3. Cấu hình đã chọn: Hybrid RRF với Dense Day 6 + TF-IDF, candidate-10, RRF-k=60, top-5. Cả hai dùng tối đa 3 lượt search và context budget 1,600 ký tự.
- **Chấm thủ công:** Đã đọc toàn bộ 23 câu dev và 7 câu test, gồm 5 câu test có đáp án và 2 câu test thiếu evidence. Citation chỉ được chấm đúng khi có trong context và hỗ trợ phát biểu tương ứng. Mỗi query, tool call, context, câu trả lời, thời gian, token usage, nhãn và lý do được lưu trong `submission/results/`.

| Cấu hình / split | Tool use | Grounded answer (0/1/2; TB /2) | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm | Provider errors | Token usage | Chi phí ước tính |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Dense / dev | 23/23 | 7 / 0 / 14; **1.33** | 15/23 | 19/19 | 2/2 | 2/21 | 0/23 | 15,039 + 2,766 = 17,805 | $0.00391545 |
| Dense / test | 7/7 | 1 / 1 / 3; **1.40** | 6/6 | 4/4 | 2/2 | 1/5 | 0/7 | 6,038 + 746 = 6,784 | $0.00135330 |
| Hybrid / dev | 23/23 | 7 / 1 / 13; **1.29** | 14/18 | 17/17 | 2/2 | 4/21 | 0/23 | 16,401 + 2,591 = 18,992 | $0.00401475 |
| Hybrid / test | 7/7 | 1 / 1 / 3; **1.40** | 5/5 | 4/4 | 2/2 | 1/5 | 0/7 | 6,733 + 714 = 7,447 | $0.00143835 |

Grounded-answer counts chỉ tính câu có đáp án; citation validity là citation có trong context và hỗ trợ câu trả lời. Số token là input + output. Giá ước tính theo standard `gpt-4o-mini`: $0.15/1M input, $0.075/1M cached input, $0.60/1M output; tổng bốn lần chạy full-split khoảng **$0.01072185**. Hai lần dense không lưu số cached token nên giả định input uncached; xem [README kết quả](../../submission/results/README.md) và [bảng giá OpenAI](https://developers.openai.com/api/docs/pricing/) để biết cách tính. Giá thực tế có thể khác hóa đơn.

- **Nhận xét lỗi:** Hybrid dev từ chối nhầm q02, q07, q13, q19; test từ chối nhầm q22. Evidence có lúc nằm trong ranking nhưng không được pack vào context cuối. Dev q08, q17, q20 trả lời sai hoặc không được citation hỗ trợ; q01 trả lời một phần. Test q23 trả lời một phần vì bỏ sót lợi ích giảm bias. Các nhãn lỗi được giữ nguyên, không chỉnh theo test.
- **Ví dụ đã rà soát:** q05 trả lời đúng với citation hỗ trợ; `test-no-evidence-01` từ chối đúng; q20 là câu trả lời không được evidence hỗ trợ.
- **Số lượt search:** Mỗi split dev có 19 câu 1 lượt và 4 câu 3 lượt; mỗi split test có 4 câu 1 lượt và 3 câu 3 lượt.
- **Validation:** 40 unit tests pass; `compileall` và `git diff --check` pass.
- **Trace agent:** `submission/results/agent-dense-{dev,test}/` và `submission/results/agent-hybrid-{dev,test}/`; mỗi thư mục có `cases.json`, `traces.json`, `reviews.json`, `metrics.json` và `run.json`.

## Thử thách nâng cao

Đã xây dựng pipeline cho *Introductory Statistics 2e* của OpenStax/Rice University (847 trang, CC BY-NC-SA 4.0), index toàn bộ tài liệu, trích bảng có cấu trúc, ghi caption/metadata hình và bổ sung mô tả thủ công đã kiểm tra cho một tập sơ đồ/biểu đồ. Bản PDF không được commit; `submission/advanced/download_source.py` tải và kiểm tra checksum.

- **Index:** 845 trang có text trích xuất, 524 bảng / 3.190 hàng, 345 figure captions, 517 image instances và 3.100 chunks. Trang 1–2 là bìa không có text. Cách chạy, nguồn, license, cấu hình và giới hạn được ghi trong [`submission/advanced/README.md`](../../submission/advanced/README.md).
- **Benchmark:** 18 câu dev và 30 câu test; test có 24 câu answerable, 6 câu unanswerable và đạt tối thiểu từng nhóm. Hash test SHA-256: `6e746d4c787919310c9b872b3af264c70b7459a7fc5a167ec424066ef593b552`; manifest ở `submission/advanced/benchmark_manifest.json`.
- **Cấu hình test:** TF-IDF với exact-reference boost cho bảng/hình được gọi tên, top-k 5, tối đa 5 lượt search, context 12.000 ký tự, `gpt-4o-mini`. Test được chạy một lượt sau khi đóng băng; lỗi provider/hệ thống: 0.
- **Chấm thủ công:** 25/30 = **83,33%**, dưới ngưỡng bonus 90% (27/30). Theo nhóm: bảng 6/6, ảnh/sơ đồ 3/5, biểu đồ 5/5, multi-hop 2/4, text 3/4, unanswerable 6/6. Năm câu không đạt: ba answerable bị abstain do citation ID thiếu/sai định dạng; hai câu multi-hop không cite đủ các nguồn được yêu cầu.
- **Chi phí và latency:** 76.851 input + 3.119 output tokens, chi phí ước tính $0.01082625; median 2.418 giây, tổng 93.538 giây. Giá là ước tính, không phải billing xác nhận.
- **Kết luận:** đã hoàn thành artifacts và đánh giá thử thách nâng cao nhưng **không đạt điều kiện 90%**, nên không khai báo đủ điều kiện nhận 15 điểm bonus. Không tuning dựa trên test.
