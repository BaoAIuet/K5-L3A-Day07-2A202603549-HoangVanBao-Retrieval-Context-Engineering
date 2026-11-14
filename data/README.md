# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Benchmark Day 6 đã chuyển đổi

Chạy `python starter/prepare_day7_data.py --day6-root "<đường-dẫn-Day-6>"` bằng Python environment có `chromadb` của Day 6 để xuất `data/day7_questions.json` và chunks cục bộ tại `starter/data/day6_chunks.json`. Script kiểm tra từng evidence có trong index, đẩy các câu test Day 6 đã xem sang dev và tạo test mới. Chunks nằm trong thư mục bị Git bỏ qua; không commit toàn bộ trích xuất nội dung sách. Các câu mới và lý do thiếu evidence cần được người nộp kiểm tra lại trước khi dùng benchmark.
