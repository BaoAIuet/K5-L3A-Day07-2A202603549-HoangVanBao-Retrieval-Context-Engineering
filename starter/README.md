# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `retrievers.py` | Hybrid retrieval kết hợp dense và TF-IDF bằng RRF |
| `retrieval_benchmark.py` | Benchmark TF-IDF, dense Day 6 và hybrid trên dev/test |
| `openai_controller.py` | Controller dùng API tương thích OpenAI để gọi search tool |
| `agent_runner.py` | Chạy agent thật và lưu trace/cases, không đưa gold vào model |
| `prepare_day7_data.py` | Export chunks và chuẩn bị benchmark từ Day 6 |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Để chuẩn bị bộ benchmark Day 7 và xuất chunks từ Chroma Day 6, chạy bằng Python environment của Day 6:

```powershell
python starter\prepare_day7_data.py --day6-root "D:\path\to\Day06"
```

Lệnh tạo `data/day7_questions.json` và `starter/data/day6_chunks.json`. Các câu test Day 6 đã xem được chuyển sang dev; bộ test mới có năm câu có evidence và hai câu thiếu evidence. Script kiểm tra evidence trước khi ghi file. Thư mục `starter/data/` bị Git bỏ qua; không nộp chunks trích xuất nếu chưa xác nhận quyền chia sẻ. Đọc và kiểm tra các câu, đặc biệt là `absence_reason`, trước khi chạy agent.

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

Để dense retrieval trực tiếp từ Chroma Day 6, dùng adapter `Day6DenseRetriever`. Adapter dùng FastEmbed `BAAI/bge-small-en-v1.5` và collection `chunks_A` hoặc `chunks_B`; hãy chạy bằng Python environment có `chromadb` và `fastembed` của Day 6:

```python
import os
from pathlib import Path
from day6_retriever import Day6DenseRetriever

day6_root = Path(os.environ["DAY6_ROOT"])
retriever = Day6DenseRetriever.from_persistent_index(
    day6_root / "starter" / "data" / "chroma_v2_b1000",
    strategy="B",
)
hits = retriever.search("your question", filters={"chapter": 13}, top_k=3)
```

Chạy đoạn trên từ thư mục `starter` của Day 7. Adapter trả về chunk theo schema Day 7 và để lỗi index, model hoặc kết nối nổi lên thay vì dùng dữ liệu giả. Hỗ trợ filter Day 6: `chapter`, `page`, `document_id`, `section`. Nếu chưa có index `chroma_v2_b1000`, chọn một thư mục Chroma đã tồn tại và đúng strategy. Converter chuẩn hóa `gold_page` dạng số hoặc danh sách thành evidence tương ứng.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 5. Benchmark retrieval

Chạy bằng Python environment Day 6 (có `chromadb` và `fastembed`), từ thư mục gốc Day 7:

```powershell
python starter\retrieval_benchmark.py --chunks starter\data\day6_chunks.json --questions data\day7_questions.json --split dev --method tfidf --top-k 3 --budget 800 --output starter\results\tfidf-dev
python starter\retrieval_benchmark.py --chunks starter\data\day6_chunks.json --questions data\day7_questions.json --split dev --method dense --day6-index "D:\path\to\Day06\starter\data\chroma_v2_b1000" --strategy B --top-k 3 --budget 800 --output starter\results\dense-dev
python starter\retrieval_benchmark.py --chunks starter\data\day6_chunks.json --questions data\day7_questions.json --split dev --method hybrid --day6-index "D:\path\to\Day06\starter\data\chroma_v2_b1000" --strategy B --candidate-k 10 --top-k 5 --budget 1600 --output starter\results\hybrid-dev
```

Hybrid dùng Reciprocal Rank Fusion (RRF) với `rrf-k=60`; thay đổi `--top-k` và `--budget` là cấu hình cải thiện, không phải phương pháp mới. Chỉ chạy test sau khi chọn cấu hình trên dev.

## 6. Chạy model thật

`OpenAICompatibleController` dùng Chat Completions tool calling. Mở `.env` ở thư mục gốc repo (nếu chưa có, sao chép `.env.example`) và điền key mới sau `OPENAI_API_KEY=`. Không thêm key vào repo, trace hoặc commit; `.env` đã được Git bỏ qua. File được nạp tự động khi chạy controller.

```powershell
notepad .env
python starter\agent_runner.py --questions data\day7_questions.json --split dev --limit 1 --day6-index "D:\path\to\Day06\starter\data\chroma_v2_b1000" --strategy B --top-k 3 --max-searches 3 --budget 1600 --output starter\results\agent-smoke
```

Nếu biến môi trường đã được đặt trong PowerShell, biến đó được ưu tiên hơn `.env`; hãy mở cửa sổ PowerShell mới nếu trước đó đã gán nhầm giá trị. Thay `--limit 1` bằng `--limit 23` hoặc bỏ `--limit` để chạy toàn split sau khi smoke test thành công. Chỉ dùng endpoint/model được phép gửi tài liệu tới; runner ghi token usage nhưng không đoán giá vì chi phí phụ thuộc provider/model. Nó tạo `cases.json` và `traces.json`; hãy đọc trace/citation và tự chấm `reviews.json` trước khi chạy `agent_metrics.py`. Không đánh giá chất lượng agent bằng mock.

## 7. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
