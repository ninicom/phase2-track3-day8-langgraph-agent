# Day 08 — LangGraph Support-Ticket Agent

Một agent phân loại & xử lý ticket hỗ trợ xây bằng **LangGraph**, có đầy đủ: phân loại
bằng LLM, định tuyến theo điều kiện, vòng lặp thử lại có giới hạn, phê duyệt
Human-in-the-Loop (HITL), lưu trạng thái (checkpointer), thu thập metrics và một
**giao diện web** để trực quan hóa hoạt động của agent.

> Trạng thái: đã hoàn thiện toàn bộ phần `nodes`, `routing`, `graph`, `persistence`,
> `report` và web UI. `make run-scenarios` đạt **7/7 kịch bản thành công**.

---

## Luồng hoạt động

```
START → intake → classify ─(route_after_classify)─┐
   simple        → answer → finalize → END
   tool          → tool → evaluate ─(route_after_evaluate)─ success → answer → finalize → END
                                                        └─ needs_retry → retry ─(route_after_retry)─┐
                                                                              tool (lặp lại)        │
                                                                              dead_letter → finalize → END
   missing_info  → clarify → finalize → END
   risky         → risky_action → approval ─(route_after_approval)─ approved → tool → evaluate → ...
                                                                  └ rejected → clarify → finalize → END
   error         → retry ─(route_after_retry)─ ...
```

- **11 node**, **4 hàm định tuyến điều kiện**. Mọi nhánh đều kết thúc tại `finalize → END`.
- Vòng lặp thử lại `tool → evaluate → retry → tool` bị chặn bởi `attempt < max_attempts`;
  khi hết lượt sẽ chuyển sang `dead_letter` (chiến lược 3 tầng: retry → fallback → dead-letter).
- `classify_node` và `answer_node` gọi LLM thật; `evaluate_node` dùng LLM-as-judge có
  fallback heuristic. Agent trả lời theo đúng ngôn ngữ của câu hỏi.

Xem sơ đồ mermaid trực tiếp ở tab **Sơ đồ** của web UI, hoặc:
`graph.get_graph().draw_mermaid()`.

---

## State & reducers

State phẳng, có kiểu (`TypedDict` + Pydantic), gọn để checkpoint nhẹ.

| Trường | Reducer | Lý do |
|---|---|---|
| `route`, `risk_level`, `attempt`, `evaluation_result` | overwrite | trạng thái hiện tại, một node ghi |
| `final_answer`, `pending_question`, `proposed_action`, `approval` | overwrite | một giá trị cuối |
| `messages`, `tool_results`, `errors`, `events` | append (`operator.add`) | lịch sử audit không được ghi đè |

---

## Cấu trúc thư mục

```
src/langgraph_agent_lab/
  state.py        # schema state, Scenario, LabEvent, initial_state
  llm.py          # đọc .env + factory LLM (Gemini/OpenAI/Anthropic/DeepSeek)
  nodes.py        # 11 node function
  routing.py      # 4 hàm định tuyến điều kiện
  graph.py        # dựng & compile StateGraph + run_to_completion (auto-resume HITL)
  persistence.py  # checkpointer Memory / SQLite (WAL)
  metrics.py      # schema metrics + tổng hợp
  report.py       # render báo cáo markdown từ metrics
  scenarios.py    # nạp scenarios.jsonl
  cli.py          # lệnh run-scenarios / validate-metrics
web/
  app.py          # web server stdlib (http.server) — không cần framework
  index.html      # UI trực quan hóa (tiếng Việt)
data/sample/scenarios.jsonl
configs/lab.yaml
outputs/metrics.json
reports/lab_report.md
```

---

## Cài đặt & cấu hình

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
pip install langchain-openai langgraph-checkpoint-sqlite
```

Tạo `.env` (dự án này dùng **DeepSeek** — API tương thích OpenAI):

```env
DEEPSEEK_API_KEY=sk-...
LLM_MODEL=deepseek-v4-flash
LANGGRAPH_INTERRUPT=true          # bật phê duyệt HITL thật
CHECKPOINTER=memory
```

`llm.py` tự đọc `.env` (parser thuần stdlib, không cần `python-dotenv`) và hỗ trợ
`GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` theo thứ tự.

> **Lưu ý DeepSeek:** proxy chạy ở "thinking mode" nên từ chối `json_schema` response_format
> và `tool_choice` cưỡng bức. Vì vậy structured output dùng `method="json_mode"` (đã cấu hình sẵn).

---

## Chạy

```bash
make run-scenarios     # chạy 7 kịch bản mẫu → outputs/metrics.json + reports/lab_report.md
make grade-local       # validate schema metrics kịch bản mẫu
make test              # pytest
make web               # mở web UI tại http://127.0.0.1:8000

# Chạy bộ grading (ẩn):
python -m langgraph_agent_lab.cli run-scenarios --config configs/grading.yaml --output outputs/metrics_grading.json

# Validate bộ grading:
python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics_grading.json
```

### Web UI (`python web/app.py`)

- **Chạy**: nhập truy vấn (hoặc chọn kịch bản mẫu), xem tuyến/rủi ro, dòng thời gian
  thực thi, kết quả công cụ và câu trả lời cuối. Nếu là hành động rủi ro, UI dừng lại
  ở nút **Phê duyệt / Từ chối** (HITL interrupt thật) rồi tiếp tục graph.
- **Sơ đồ**: render mermaid của graph đã compile.
- **.env**: hiển thị cấu hình (giá trị bí mật được che).
- **Số liệu**: bảng từ `outputs/metrics.json`.

---

## Kịch bản & kết quả

`data/sample/scenarios.jsonl` gồm 7 kịch bản phủ 6 tình huống chuẩn (simple, tool,
missing-info, risky/HITL, transient-error→retry, max-error→dead-letter).

| Scenario | Expected | Actual | Success | Retries | Interrupts | Appr. req | Appr. seen |
|---|---|---|:--:|--:|--:|:--:|:--:|
| S01_simple | simple | simple | ✅ | 0 | 0 | no | no |
| S02_tool | tool | tool | ✅ | 0 | 0 | no | no |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | no | no |
| S04_risky | risky | risky | ✅ | 0 | 1 | yes | yes |
| S05_error | error | error | ✅ | 1 | 0 | no | no |
| S06_delete | risky | risky | ✅ | 0 | 1 | yes | yes |
| S07_dead_letter | error | error | ✅ | 1 | 0 | no | no |

**Tỉ lệ thành công: 100%.** Routing dựa trên phân loại LLM + logic state, không hard-code
theo ID kịch bản. (Số lần thử lại của kịch bản lỗi S05_error giảm từ 2 xuống 1 do đã bỏ phần giả lập lỗi công cụ tạm thời).

---

## Persistence & HITL

- Checkpointer lưu snapshot sau mỗi super-step; mỗi lần chạy có `thread_id` riêng.
- **SQLite** (`SqliteSaver` + WAL): `build_checkpointer("sqlite", "outputs/checkpoints.sqlite")`
  cho phép resume sau khi process khởi động lại và time-travel qua `get_state_history()`.
- **HITL**: khi `LANGGRAPH_INTERRUPT=true`, `approval_node` gọi `interrupt()`. CLI tự
  approve qua `run_to_completion`; web UI để người dùng bấm Phê duyệt/Từ chối.

---

## Kiểm thử

```bash
make test
```

19 test (routing/state/metrics) pass. 6 smoke test gọi LLM bị skip vì chúng chỉ kích hoạt
khi có `OPENAI/GEMINI/ANTHROPIC` key — luồng end-to-end đã được `run-scenarios` bao phủ.
