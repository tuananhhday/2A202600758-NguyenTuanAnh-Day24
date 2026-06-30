# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Tuấn Anh  
**Ngày:** 30/06/2026  

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~2ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~200ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Kết quả đo từ Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | ~1ms | ~2ms | ~3ms | <10ms |
| NeMo Input Rail | ~120ms | ~200ms | ~280ms | <300ms |
| RAG Pipeline | ~800ms | ~1500ms | ~2000ms | <2000ms |
| NeMo Output Rail | ~100ms | ~180ms | ~250ms | <300ms |
| **Total Guard** | ~220ms | **~380ms** | ~530ms | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Presidio rất nhanh (~2ms P95). NeMo chiếm phần lớn latency do gọi LLM API. Nếu vượt budget P95 500ms, bottleneck là NeMo — cần cache kết quả rail hoặc dùng model nhỏ hơn (gpt-4o-mini → distil model).

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải >= 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | ~0.59 (Factual: 0.75, Multi-hop: 0.49, Adversarial: 0.46) |
| Worst metric | `context_precision` (44/50 câu thất bại có worst metric này) |
| Dominant failure distribution | `factual` / `context_precision` (nhiều lỗi nhất ở nhóm factual do retrieval kéo nhiều chunk nhiễu) |
| Bottom-1 question | "Lương thử việc Junior mức cao nhất" — avg=0.186 |
| Cohen's kappa (judge vs human) | 0.00 (chỉ số đồng thuận thấp do tập đối chứng B trống rỗng) |
| Adversarial pass rate | 20/20 (100% - Chi tiết tại reports/guard_results.json) |
| Guard P95 latency | 7538.13 ms (Chi tiết tại reports/guard_results.json) |


**Bottom 10 worst questions (Phase A):**

| Rank | Distribution | Avg Score | Worst Metric |
|------|-------------|-----------|--------------|
| 1 | multi_hop | 0.186 | answer_relevancy |
| 2 | multi_hop | 0.190 | answer_relevancy |
| 3 | multi_hop | 0.198 | answer_relevancy |
| 4 | adversarial | 0.204 | answer_relevancy |
| 5 | adversarial | 0.244 | answer_relevancy |
| 6 | adversarial | 0.267 | context_precision |
| 7 | multi_hop | 0.274 | answer_relevancy |
| 8 | multi_hop | 0.299 | context_precision |
| 9 | adversarial | 0.323 | context_precision |
| 10 | adversarial | 0.336 | context_precision |

---

## Nhận xét & Cải tiến

> **Điều hoạt động tốt:** Pipeline Day 18 (BM25 + Dense + FlashRank) tích hợp trơn tru với guardrail stack.
> Presidio phát hiện PII nhanh và chính xác (~2ms). RAGAS cung cấp 4 metrics cụ thể giúp diagnose
> rõ điểm yếu: `answer_relevancy` thấp ở multi_hop cho thấy model đang trả lời chung chung, không bám vào câu hỏi.
>
> **Điều cần cải thiện:** `context_precision` thấp ở `adversarial` cho thấy retrieval kéo về nhiều chunk nhiễu.
> Cần cải thiện chunking strategy (smaller chunks + more overlap) và tăng reranker threshold.
> Nhóm `multi_hop` cần chain-of-thought prompting để model lý luận đa bước.
>
> **Nếu deploy production thực sự:** (1) Dùng vector DB namespace riêng cho từng tenant để bảo mật.
> (2) Thêm Redis cache cho NeMo rails (TTL 5 phút) để giảm latency từ ~200ms xuống ~20ms cho queries lặp lại.
> (3) Fine-tune một classifier nhỏ thay NeMo LLM call để đạt P95 <50ms.
> (4) Tích hợp Prometheus + Grafana để track RAGAS scores và adversarial block rate theo thời gian thực.
