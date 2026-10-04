# SmartShrimp RAG Service

Microservice FastAPI cung cấp chức năng hỏi đáp kỹ thuật nuôi tôm bằng
Retrieval-Augmented Generation (RAG). Service đọc tài liệu chuyên môn, chia tài liệu
theo ngữ nghĩa, lưu embedding trong PostgreSQL/pgvector và sử dụng Gemini để tạo câu
trả lời dựa trên các đoạn tài liệu tìm được.

README được chia thành hai phần:

1. [Sử dụng RAG như một service độc lập](#phần-1--sử-dụng-rag-như-một-service-độc-lập)
2. [Tích hợp RAG với hệ thống SmartShrimp](#phần-2--tích-hợp-với-hệ-thống-smartshrimp)

> **Trạng thái tích hợp:** API của RAG service đã sẵn sàng, nhưng `smartshrimp_be`
> hiện chưa có route, controller hoặc service gọi sang RAG. Phần 2 mô tả contract và
> các bước cần triển khai ở backend SmartShrimp.

## Tổng quan kiến trúc

```text
Tài liệu PDF/DOCX/MD/TXT
          │
          ▼
Loader → Semantic chunking → Gemini Embedding
                                  │
                                  ▼
                         PostgreSQL + pgvector
                                  │
                                  ▼
Câu hỏi → Query embedding → Top-K chunks → Gemini LLM → Câu trả lời
```

### Công nghệ

| Thành phần | Công nghệ |
|---|---|
| API | FastAPI, Uvicorn |
| Embedding | Google Gemini Embedding, vector 768 chiều |
| Sinh câu trả lời | Google Gemini |
| Vector store | PostgreSQL với extension `pgvector` |
| Database client | `asyncpg` |
| Đọc tài liệu | `pypdf`, `python-docx` |
| Chunking | Semantic chunking; recursive character splitting là chế độ dự phòng |
| Kiểm thử | `pytest`, `pytest-asyncio`, `httpx` |

---

# Phần 1 — Sử dụng RAG như một service độc lập

## 1. Yêu cầu

- Python 3.11 được khuyến nghị.
- PostgreSQL có extension `pgvector`.
- Gemini API key.
- Tài liệu kiến thức ở định dạng `.pdf`, `.docx`, `.doc`, `.md` hoặc `.txt`.

## 2. Cài đặt

Từ thư mục `smartshrimp_rag`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Trên macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Cấu hình môi trường

Sao chép file mẫu:

```powershell
Copy-Item .env.example .env
```

Cập nhật tối thiểu các biến sau:

```env
INTERNAL_API_KEY=replace-with-a-long-random-secret
GEMINI_API_KEY=your-gemini-api-key
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
GEMINI_LLM_MODEL=gemini-3.6-flash
DATABASE_URL=postgresql://postgres:password@localhost:5432/smartshrimp
```

`DATABASE_URL` chấp nhận cả tiền tố `postgresql://` và
`postgresql+asyncpg://`.

### Các biến cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `APP_ENV` | `development` | Môi trường chạy. Swagger bị tắt khi đặt là `production`. |
| `APP_HOST` | `0.0.0.0` | Host cấu hình của ứng dụng. |
| `APP_PORT` | `8000` | Cổng cấu hình của ứng dụng. |
| `APP_VERSION` | `1.0.0` | Phiên bản trả về ở health check. |
| `INTERNAL_API_KEY` | `change-me-in-production` | Secret bảo vệ `POST /v1/chat`. Bắt buộc đổi ngoài local. |
| `GEMINI_API_KEY` | rỗng | API key dùng cho embedding và generation. |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-2` | Model embedding. |
| `GEMINI_LLM_MODEL` | `gemini-3.6-flash` | Model sinh câu trả lời. |
| `DATABASE_URL` | rỗng | Chuỗi kết nối PostgreSQL. |
| `SIMILARITY_THRESHOLD` | `0.5` | Ngưỡng similarity tối thiểu để gọi LLM. |
| `TOP_K_DEFAULT` | `5` | Số chunk được retrieve mặc định. |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` hoặc `recursive`. |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `90` | Mức percentile dùng để phát hiện điểm chuyển chủ đề. |
| `MIN_CHUNK_SIZE` | `350` | Kích thước mục tiêu tối thiểu trước khi ngắt theo ngữ nghĩa. |
| `MAX_CHUNK_SIZE` | `1200` | Kích thước tối đa của chunk. |
| `CHUNK_OVERLAP` | `150` | Số ký tự tối đa của các câu hoàn chỉnh được gối sang chunk tiếp theo. |
| `CHUNK_SIZE` | `800` | Chỉ dùng khi `CHUNKING_STRATEGY=recursive`. |
| `MAX_CONTEXT_TOKENS` | `1048576` | Giới hạn context window của model (tokens). Prompt sẽ được cắt chunk nếu vượt. |
| `MAX_OUTPUT_TOKENS` | `8192` | Số token tối đa cho câu trả lời. Tăng nếu câu trả lời bị cụt. |
| `LOG_LEVEL` | `DEBUG` | Mức log của service. |

## 4. Khởi tạo vector store

Chạy migration sau trên PostgreSQL/Supabase SQL Editor:

```text
migrations/001_create_document_chunks.sql
```

Migration sẽ:

- bật extension `vector`;
- tạo bảng `document_chunks`;
- tạo vector column 768 chiều;
- tạo IVFFlat index dùng cosine distance;
- tạo unique constraint `(source_file, chunk_index)`.

## 5. Nạp tài liệu

Đặt tài liệu vào thư mục:

```text
data/documents/
```

Nạp một file:

```powershell
python -m app.ingestion.ingest --path "data/documents/tai-lieu.pdf"
```

Nạp toàn bộ file nằm trực tiếp trong một thư mục:

```powershell
python -m app.ingestion.ingest --path data/documents
```

Quá trình ingest thực hiện:

1. Đọc nội dung tài liệu.
2. Chia thành các semantic unit nhỏ.
3. Embed theo batch để tìm điểm chuyển chủ đề.
4. Ghép thành chunk trong giới hạn `MIN_CHUNK_SIZE` và `MAX_CHUNK_SIZE`.
5. Embed chunk hoàn chỉnh.
6. Upsert chunk vào `document_chunks`.

> Loader thư mục hiện chỉ đọc các file nằm trực tiếp trong thư mục được truyền vào,
> không tự duyệt thư mục con. Muốn ingest tài liệu trong thư mục con, hãy gọi CLI với
> đúng thư mục đó hoặc truyền đường dẫn từng file.

> Khi thay đổi thuật toán hoặc cấu hình chunking, nên xóa vector cũ trước khi ingest
> lại để không giữ các chunk dư từ lần ingest trước.

## 6. Chạy service

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Các URL trong môi trường development:

- Health check: `http://localhost:8000/v1/health`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Kiểm tra health:

```powershell
Invoke-RestMethod http://localhost:8000/v1/health
```

## 7. Gọi API chat trực tiếp

Endpoint:

```http
POST /v1/chat
X-Internal-Key: <INTERNAL_API_KEY>
Content-Type: application/json
```

Ví dụ PowerShell:

```powershell
$headers = @{ "X-Internal-Key" = "replace-with-your-secret" }
$body = @{
    question = "Cách xử lý khi pH ao nuôi biến động mạnh?"
    user_id = "00000000-0000-0000-0000-000000000001"
    conversation_id = $null
    season_id = $null
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/v1/chat `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
```

Request body:

| Trường | Kiểu | Bắt buộc | Ghi chú |
|---|---|---:|---|
| `question` | string | Có | Từ 1 đến 2000 ký tự. |
| `user_id` | UUID | Có | ID người đặt câu hỏi; hiện dùng để log. |
| `conversation_id` | UUID hoặc `null` | Không | ID hội thoại; hiện dùng để log. |
| `season_id` | UUID hoặc `null` | Không | ID vụ nuôi; hiện chưa dùng để lọc retrieval. |
| `top_k` | integer hoặc `null` | Không | Từ 1 đến 20; bỏ trống để dùng `TOP_K_DEFAULT`. |

Response mẫu:

```json
{
  "answer": "Cần kiểm tra độ kiềm, mật độ tảo và đo pH vào sáng sớm, buổi chiều...",
  "query_status": "answered",
  "retrieved_chunks": [
    {
      "source_file": "huong-dan-nuoi-tom.pdf",
      "chunk_index": 4,
      "content": "Nội dung đoạn tài liệu được sử dụng...",
      "similarity": 0.82,
      "metadata": {
        "format": "pdf",
        "pages": 20,
        "chunking_strategy": "semantic",
        "chunk_size_chars": 934
      }
    }
  ],
  "top_similarity": 0.82,
  "model_version": "gemini-3.6-flash",
  "processing_time_ms": 2100
}
```

### `query_status`

| Trạng thái | Ý nghĩa |
|---|---|
| `answered` | Có chunk đạt ngưỡng và Gemini đã sinh câu trả lời. |
| `low_match` | Similarity thấp hơn ngưỡng; vẫn sinh câu trả lời, backend kèm cảnh báo cần kiểm chứng. |
| `no_source` | Không có chunk; vẫn sinh câu trả lời tổng quát, backend cảnh báo chưa có nguồn tham chiếu. |
| `error` | Lỗi embedding, database, Gemini hoặc lỗi kỹ thuật khác. |

### Mã HTTP đáng chú ý

| HTTP status | Trường hợp |
|---:|---|
| `200` | Pipeline chạy xong; kết quả nghiệp vụ nằm trong `query_status`. |
| `401` | `X-Internal-Key` sai. |
| `422` | Thiếu header, UUID không hợp lệ, câu hỏi rỗng hoặc `top_k` ngoài khoảng. |
| `500` | Lỗi chưa được pipeline xử lý. |

## 8. Chạy kiểm thử

```powershell
pytest tests -v
```

Các test semantic chunking sử dụng embedding giả, không gọi Gemini thật.

## 9. Chạy bằng Docker

```powershell
docker compose up --build -d
```

Xem log:

```powershell
docker compose logs -f rag
```

Dừng service:

```powershell
docker compose down
```

> Healthcheck trong `docker-compose.yml` đang dùng lệnh `curl`, nhưng Docker image
> hiện chưa cài `curl`. Container có thể phục vụ bình thường nhưng bị Docker đánh dấu
> `unhealthy`; cần bổ sung `curl` vào image hoặc đổi healthcheck sang Python trước khi
> dùng cấu hình này ở production.

## 10. Cấu trúc thư mục

```text
smartshrimp_rag/
├── app/
│   ├── api/v1/endpoints/     # Health và chat endpoints
│   ├── core/                 # Embedder, retriever, generator, pipeline
│   ├── db/                   # PostgreSQL pool và vector store
│   ├── ingestion/            # Loader, semantic chunker, ingest CLI
│   ├── models/               # Request/response schemas
│   ├── config.py             # Environment settings
│   ├── dependencies.py       # X-Internal-Key guard
│   └── main.py               # FastAPI application
├── data/documents/           # Tài liệu nguồn, không commit
├── migrations/               # Migration cho document_chunks
├── tests/                    # Unit và API tests
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

# Phần 2 — Tích hợp với hệ thống SmartShrimp

## 1. Phân chia trách nhiệm

```text
Mobile/Web App
     │ JWT của người dùng
     ▼
smartshrimp_be
     │ xác thực, phân quyền, quản lý hội thoại, lưu lịch sử
     │ X-Internal-Key
     ▼
smartshrimp_rag
     │ embed → retrieve → generate
     ▼
PostgreSQL/pgvector + Gemini
```

### `smartshrimp_be` chịu trách nhiệm

- Xác thực JWT của người dùng.
- Kiểm tra người dùng có quyền truy cập `season_id` hay không.
- Tạo hoặc kiểm tra `rag_conversations`.
- Gọi RAG service bằng internal key.
- Lưu request và kết quả vào `rag_queries`.
- Trả response phù hợp cho mobile/web.
- Quản lý feedback trong `rag_feedback`.

### `smartshrimp_rag` chịu trách nhiệm

- Xác thực `X-Internal-Key` giữa hai service.
- Embed câu hỏi.
- Tìm top-K chunk trong `document_chunks`.
- Kiểm tra similarity threshold.
- Sinh câu trả lời dựa trên tài liệu; khi thiếu nguồn, trả lời tổng quát và yêu cầu kiểm chứng.
- Trả kết quả retrieval và thông tin xử lý.

RAG service không xác thực JWT và không nên được mobile/web gọi trực tiếp.

## 2. Cấu hình giữa hai service

Hai service phải dùng cùng một internal key.

Trong `smartshrimp_rag/.env`:

```env
INTERNAL_API_KEY=a-long-random-shared-secret
```

Các biến sau cần được bổ sung vào `smartshrimp_be/.env` khi triển khai connector:

```env
RAG_SERVICE_URL=http://localhost:8000
RAG_INTERNAL_API_KEY=a-long-random-shared-secret
RAG_REQUEST_TIMEOUT_MS=60000
```

Khi chạy hai service trong cùng Docker network, URL nên dùng service name, ví dụ:

```env
RAG_SERVICE_URL=http://rag:8000
```

Không đưa `RAG_INTERNAL_API_KEY` vào source code, mobile app hoặc frontend.

## 3. Contract gọi từ backend sang RAG

Backend gọi:

```http
POST {RAG_SERVICE_URL}/v1/chat
X-Internal-Key: {RAG_INTERNAL_API_KEY}
Content-Type: application/json
```

Body:

```json
{
  "question": "Dấu hiệu bệnh đục cơ trên tôm là gì?",
  "user_id": "0e563d8b-a3f2-43c9-af85-7b3b58221230",
  "conversation_id": "2c810541-4744-4767-b9dd-6ed05a4f15d5",
  "season_id": null,
  "top_k": 5
}
```

Ví dụ connector dùng `axios` trong `smartshrimp_be`:

```js
const axios = require('axios');

async function askRag(payload) {
    const baseUrl = process.env.RAG_SERVICE_URL.replace(/\/+$/, '');
    const timeout = Number(process.env.RAG_REQUEST_TIMEOUT_MS || 60000);

    const response = await axios.post(`${baseUrl}/v1/chat`, payload, {
        timeout,
        headers: {
            'Content-Type': 'application/json',
            'X-Internal-Key': process.env.RAG_INTERNAL_API_KEY,
        },
    });

    return response.data;
}
```

Đây là ví dụ contract, chưa phải file đang tồn tại trong `smartshrimp_be`.

## 4. Luồng xử lý đề xuất ở backend

1. Client gửi câu hỏi tới endpoint chat của `smartshrimp_be` kèm JWT.
2. Backend lấy `account_id` từ access token; không tin `user_id` do client gửi.
3. Nếu có `season_id`, backend kiểm tra quyền truy cập vụ nuôi.
4. Backend tạo hội thoại mới hoặc xác minh `conversation_id` thuộc người dùng.
5. Backend gọi `POST /v1/chat` của RAG service.
6. Backend lưu kết quả vào `rag_queries`.
7. Backend cập nhật `rag_conversations.last_message_at`.
8. Backend trả câu trả lời và `query_id` cho client để có thể gửi feedback.

Không để client tự truyền internal key hoặc gọi thẳng cổng `8000`.

## 5. Ánh xạ response sang database SmartShrimp

Schema SQL hiện có các bảng `rag_conversations`, `rag_queries` và `rag_feedback`.
Kết quả RAG có thể ánh xạ vào `rag_queries` như sau:

| Response RAG | Cột `rag_queries` |
|---|---|
| request `conversation_id` | `conversation_id` |
| request `user_id` | `account_id` |
| request `question` | `question` |
| `answer` | `answer` |
| `query_status` | `query_status` |
| `retrieved_chunks` | `retrieved_chunks` |
| `top_similarity` | `top_similarity` |
| `model_version` | `model_version` |
| `processing_time_ms` | `processing_time_ms` |

Lưu ý khi tích hợp:

- Nhánh `feat/85-87-rag-conversations-feedback` của backend đã bổ sung Prisma models
  ánh xạ các bảng RAG hiện có trong `smartshrimp.sql`. Chạy `npx prisma generate`
  trước khi khởi động backend; không cần thay đổi database.
- Constraint của `rag_queries` yêu cầu `error_message` khi `query_status='error'`,
  trong khi `ChatResponse` hiện không trả `error_message`. Backend cần tạm lưu một
  thông báo lỗi chuẩn hóa, hoặc contract RAG cần được mở rộng trước khi hoàn thiện
  nhánh lỗi.
- `user_id`, `conversation_id` và `season_id` hiện chỉ được RAG service ghi log;
  retrieval chưa lọc theo người dùng, hội thoại hoặc vụ nuôi.

## 6. Xử lý lỗi khi tích hợp

Backend nên phân biệt hai nhóm lỗi:

### RAG trả HTTP 200

Đọc `query_status`:

- `answered`: lưu và trả câu trả lời.
- `low_match`: lưu và trả câu trả lời cùng nguồn đã retrieve; cảnh báo độ liên quan thấp.
- `no_source`: lưu và trả câu trả lời tổng quát; cảnh báo chưa có nguồn tham chiếu.
- `error`: lưu lỗi chuẩn hóa; trả thông báo tạm thời không thể xử lý.

### RAG không trả HTTP 200

- `401`: cấu hình internal key giữa hai service không khớp.
- `422`: payload backend gửi sang RAG không hợp lệ.
- timeout, mất kết nối hoặc `5xx`: ghi log kỹ thuật ở backend và trả lỗi dịch vụ
  tạm thời cho client.

Không trả Gemini API key, internal key, stack trace hoặc chi tiết kết nối database
cho client.

## 7. Checklist tích hợp SmartShrimp

Backend API và Flutter đã được triển khai trên nhánh `feat/85-87-rag-conversations-feedback`
trong từng repository. Xem `smartshrimp_be/docs/rag-api.md` để biết contract, quyền,
cấu hình và kết quả kiểm thử. Checklist dưới đây là hướng dẫn triển khai môi trường thực.

- [ ] Thêm cấu hình RAG vào `smartshrimp_be/src/config/index.js`.
- [ ] Thêm service gọi `POST /v1/chat` bằng `axios`.
- [ ] Thêm validator cho câu hỏi, `conversation_id`, `season_id` và `top_k`.
- [ ] Thêm route/controller chat có middleware xác thực người dùng.
- [ ] Thêm Prisma models hoặc data-access cho các bảng RAG.
- [ ] Kiểm tra quyền sở hữu conversation và quyền truy cập season.
- [ ] Lưu mọi trạng thái `answered`, `low_match`, `no_source`, `error`.
- [ ] Thêm endpoint gửi feedback vào `rag_feedback`.
- [ ] Không public trực tiếp RAG service ra mobile/web.
- [ ] Thêm integration test mock RAG service và test timeout/lỗi kết nối.

## 8. Kiểm tra sau khi tích hợp

1. Gọi `GET /v1/health` từ môi trường chạy backend.
2. Gọi RAG trực tiếp bằng internal key để xác nhận contract.
3. Gọi endpoint chat của `smartshrimp_be` bằng JWT hợp lệ.
4. Kiểm tra bản ghi `rag_conversations` và `rag_queries`.
5. Kiểm tra các nhánh `low_match`, `no_source`, `error`.
6. Xác nhận mobile/web không biết và không gửi `X-Internal-Key`.

