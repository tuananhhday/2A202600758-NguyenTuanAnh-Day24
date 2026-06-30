# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Tuấn Anh  
**Ngày:** 30/06/2026  
**Judge model:** gpt-4o-mini  

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên các cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên nghỉ bao nhiêu ngày khi kết hôn? | A | A cung cấp thông tin số ngày nghỉ có lương chính xác, B không có thông tin. |
| 2 | Mua thiết bị 55 triệu cần ai phê duyệt? | A | A cung cấp thông tin người phê duyệt cụ thể, B trống thông tin. |
| 3 | Thưởng Tết tối thiểu cho NV chính thức? | A | A cung cấp đúng mức tối thiểu 1 tháng lương, B trống thông tin. |
| 4 | Senior thâm niên 9 năm: số ngày phép & lương? | A | A cung cấp đủ số phép tích lũy và khoảng lương, B không có thông tin. |
| 5 | Hoàn trả tiền tài trợ khóa học khi nghỉ việc? | A | A giải thích đúng quy định hoàn trả 100%, B không có thông tin. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | A | A | A | Yes |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | A | A | A | Yes |

**Position bias rate:** 0.0% (Tất cả 11 câu hỏi chạy thử đều có kết quả nhất quán khi đổi vị trí)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)  
**Judge labels:** [kết quả chạy judge trên 10 câu tương ứng]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 1 | No |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 1 | No |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 1 | No |

**Cohen's κ:** 0.0  
**Interpretation:** Poor agreement (Thấp vì Judge gán nhãn 1 cho toàn bộ do câu trả lời đối sánh B hoàn toàn trống rỗng).

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 11 / 11 cases
- B thắng + B dài hơn A: 0 / 11 cases  
- **Verbosity bias rate:** 100%

**Kết luận:** 
Mô hình LLM có xu hướng rất mạnh trong việc lựa chọn câu trả lời dài hơn và nhiều thông tin hơn (Verbosity Bias), đặc biệt khi câu trả lời đối trọng quá ngắn hoặc không có thông tin. Trong môi trường thực tế, đây là một vấn đề nghiêm trọng vì mô hình có thể ưu tiên một câu trả lời dài dòng chứa thông tin sai lệch/lỗi thời hơn là một câu trả lời ngắn gọn nhưng chính xác.

---

## 5. Nhận xét chung

- Chỉ số Cohen's Kappa đạt 0.0 cho thấy mức độ tương đồng giữa Judge và con người là rất thấp khi sử dụng một baseline đối chứng trống rỗng ("Không có thông tin"). Để cải thiện chỉ số này lên >0.6, cần thiết kế câu trả lời đối chứng B là câu trả lời có chất lượng tương đương hoặc so sánh trực tiếp với Ground Truth.
- Position bias đạt 0.0% trong bài thử nghiệm này do sự chênh lệch chất lượng quá lớn giữa A và B. Tuy nhiên, trong production khi chất lượng hai câu trả lời A/B xấp xỉ nhau, tỷ lệ này thường tăng cao (>20%).
- Kỹ thuật **Swap-and-Average** thực sự hữu ích vì nó bắt buộc mô hình phải đánh giá nội dung một cách độc lập với thứ tự trình bày, loại bỏ hoàn toàn các lỗi đưa ra quyết định dựa trên thói quen vị trí.
- Trong production, nên dùng LLM Judge kết hợp với Reranking và RAGAS metrics để có cái nhìn toàn diện, đồng thời chạy swap liên tục để đảm bảo tính khách quan cho kết quả đánh giá tự động.
