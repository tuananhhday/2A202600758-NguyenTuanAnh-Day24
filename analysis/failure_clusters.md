# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Tuấn Anh  
**Ngày:** 30/06/2026  

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.8947 | 0.6000 | 0.6663 |
| answer_relevancy | 0.9233 | 0.5175 | 0.5308 |
| context_precision | 0.2778 | 0.2677 | 0.2272 |
| context_recall | 0.9073 | 0.5939 | 0.4253 |
| **avg_score** | 0.7508 | 0.4948 | 0.4624 |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.1862 | answer_relevancy |
| 2 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.1900 | answer_relevancy |
| 3 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.1975 | answer_relevancy |
| 4 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.2036 | answer_relevancy |
| 5 | adversarial | Theo chính sách nghỉ phép cũ (v2023), nhân viên được nghỉ bao nhiêu ngày? Còn chính sách nào đang có hiệu lực hiện tại? | 0.2443 | answer_relevancy |
| 6 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.2673 | context_precision |
| 7 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.2741 | answer_relevancy |
| 8 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.2988 | context_precision |
| 9 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.3228 | context_precision |
| 10 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.3363 | context_precision |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 0 | 4 | 2 | 6 |
| context_precision | 20 | 16 | 8 | 44 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (tập trung nhiều failure đếm được về context_precision)  
**Dominant metric:** context_precision  

**Lý do phân tích:**

`context_precision` là điểm yếu lớn nhất của hệ thống trên cả 3 distributions (tổng cộng 44/50 câu có worst metric là context_precision). Nguyên nhân chủ yếu do quá trình retrieval đang kéo về quá nhiều chunk nhiễu hoặc không thực sự liên quan sát sườn đến câu hỏi. Đối với các câu hỏi `factual` đơn giản, mặc dù mô hình vẫn sinh ra được câu trả lời đúng (thể hiện qua faithfulness và recall cao), nhưng do tỷ lệ chunk nhiễu trong top-k trả về cao nên làm giảm nghiêm trọng điểm số precision của ngữ cảnh.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM tự suy diễn hoặc hallucinate khi context không có thông tin rõ ràng | Cải thiện prompt sinh câu trả lời bằng cách bắt buộc LLM chỉ trích dẫn trực tiếp từ ngữ cảnh. |
| context_recall | Bỏ sót các chunk chứa thông tin cốt lõi do tìm kiếm từ vựng (lexical) kém | Tích hợp tìm kiếm Hybrid (kết hợp Dense và Sparse/BM25) và áp dụng chunk enrichment như Contextual Prepending. |
| context_precision | Thứ tự các chunk trả về chưa tối ưu, chứa quá nhiều chunk nhiễu | Áp dụng mô hình Rerank (như FlashRank/CrossEncoder) và nâng cao ngưỡng lọc điểm số của vector để loại bỏ các chunk kém chất lượng. |
| answer_relevancy | LLM sinh câu trả lời dài dòng hoặc đi lạc đề so với câu hỏi phức tạp | Áp dụng kỹ thuật Chain-of-Thought (CoT) prompting và tinh chỉnh câu lệnh hệ thống để LLM tập trung trả lời đúng trọng tâm. |

---

## 6. Nhận xét về Adversarial Distribution

Điểm trung bình (avg_score) của nhóm `adversarial` là **0.4624**, thấp nhất trong cả 3 nhóm (so với `factual` là **0.7508** và `multi_hop` là **0.4948**). Điều này cho thấy pipeline RAG hiện tại rất dễ bị đánh lừa bởi các bẫy phủ định và bẫy phiên bản chính sách cũ/mới (v2023 vs v2024). 

Trong bottom 10, có tới 5 câu thuộc nhóm `adversarial` (như các câu xếp hạng 4, 5, 6, 9, 10). Điển hình là câu hỏi về việc dùng VPN cá nhân hoặc câu hỏi về số ngày phép năm theo phiên bản cũ. Lý do là vì hệ thống RAG truy xuất ra cả tài liệu cũ lẫn mới và LLM không phân biệt được đâu là tài liệu đang có hiệu lực pháp lý cao nhất, dẫn tới việc tổng hợp sai thông tin.
