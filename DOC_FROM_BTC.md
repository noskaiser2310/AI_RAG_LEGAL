# Statute Law Retrieval & Question Answering

**Tác giả:** Nguyễn Hoàng Trung | Nguyễn Tấn Minh  
**Đơn vị:** Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội (VNU-UET) | Viện Khoa học và Công nghệ Tiên tiến Nhật Bản (JAIST)  
**Ngày:** 12 tháng 06 năm 2026  

## Mục lục
1. Giới thiệu bài toán
2. Dữ liệu
3. Các phương pháp đánh giá
4. Tổng thể kiến trúc Retrieval và Question Answering

---

## 1. Giới thiệu bài toán

### Hai hệ thống pháp luật chính
| Tiêu chí | Common Law (Thông luật) | Civil Law (Dân luật / Luật thành văn) |
|---|---|---|
| **Nguồn luật** | Luật án lệ | Luật thành văn |
| **Tại sao quan trọng với AI?** | Cấu trúc dữ liệu quyết định bài toán Retrieval, metric đánh giá, và cả việc lựa chọn chuyên gia gán nhãn. |

### Common Law (Thông luật)
* **Định nghĩa:** Hệ thống pháp luật mà nguồn luật chủ yếu là án lệ (case law) do tòa án tạo ra qua các phán quyết; thẩm phán giải thích và phát triển luật từ tiền lệ. Tiêu biểu: Anh, Mỹ, Canada, Úc.
* **Đặc điểm chính:**
  * *Nguyên tắc:* Stare decisis — phán quyết trước ràng buộc phán quyết sau.
  * *Nguồn luật chính:* Bản án tòa án (judicial decisions).
  * *Phương pháp:* Tư duy quy nạp (inductive).
* **Bài toán AI Case Retrieval:**
  * Tìm án lệ tương tự.
  * Dựa trên sự kiện, luận điểm.
  * Precedent matching.

### Civil Law (Dân luật / Luật thành văn)
* **Định nghĩa:** Hệ thống pháp luật dựa trên văn bản pháp luật thành văn được hệ thống hóa (bộ luật); thẩm phán áp dụng luật chứ không tạo luật, án lệ chỉ mang tính tham khảo. Tiêu biểu: Lục địa châu Âu, Nhật Bản, Việt Nam.
* **Đặc điểm chính:**
  * *Nguyên tắc:* Bộ luật là nguồn chính; án lệ chỉ mang tính tham khảo.
  * *Nguồn luật chính:* Văn bản quy phạm pháp luật.
  * *Phương pháp:* Tư duy diễn dịch (deductive).
* **Bài toán AI Statute Retrieval:**
  * Ánh xạ tình huống → điều luật.
  * Cấu trúc phân cấp rõ ràng.
  * Cross-domain relevance.

### Bài toán 1: Statute Retrieval (SLR)
* **Đầu vào (Q):** Câu hỏi hoặc tình huống thực tế mô tả sự kiện pháp lý cần tra cứu.
* **Kho dữ liệu (D):** Toàn bộ hệ thống văn bản pháp luật đã được số hóa và cấu trúc hóa.
* **Nhiệm vụ:** Từ D, tìm và xếp hạng các văn bản/điều luật liên quan nhất đến Q.
* **Đầu ra:** Danh sách xếp hạng các điều luật/ văn bản pháp lý.
* *Luồng:* `Q` → `Retrieval System` → `Điều luật`

#### Ví dụ: Statute Retrieval
* **Câu hỏi (Q):** "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
* **Đầu ra mong đợi:**
  * Luật 04/2017/QH14 | Điều 4
  * Luật 04/2017/QH14 | Điều 5
  * Nghị định 80/2021/NĐ-CP | Điều 5
* *Lưu ý quan trọng:* Hệ thống phải trả về đúng văn bản cụ thể và điều luật cụ thể, không chỉ tên luật chung chung.

### Bài toán 2: Statute QA
* **Đầu vào (Q):** Một câu hỏi pháp lý cần được trả lời.
* **Điều luật (D):** Các điều luật đã được truy xuất từ bài toán 1 (không cần tìm lại).
* **Nhiệm vụ:** Dựa trên điều luật đã cho, đưa ra câu trả lời chính xác.
* **Đầu ra (A):** Yes/No/ Trích xuất/ Suy luận.
* *Luồng:* `Q` + `D` → `QA System` → `Trả lời`

#### Ví dụ: Statute QA
* **Câu hỏi (Q):** "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?"
* **Điều luật được cung cấp (D):**
  * Luật 04/2017/QH14 | Điều 4 — Phạm vi điều chỉnh
  * Luật 04/2017/QH14 | Điều 5 — Đối tượng áp dụng
  * Nghị định 80/2021/NĐ-CP | Điều 5 — Tiêu chí DNNVV
* **Trả lời (A):** Doanh nghiệp được thành lập theo pháp luật, đáp ứng tiêu chí DNNVV (không quá 200 lao động, vốn không quá 100 tỷ đồng hoặc doanh thu không quá 300 tỷ đồng), và thực hiện đầy đủ nghĩa vụ theo Luật.

---

## 2. Dữ liệu

### Nguồn dữ liệu pháp luật Việt Nam
* **Nguồn thu thập:**
  * *Chính thức (Nhà nước):* vbpl.vn (CSDL Quốc gia - Bộ Tư pháp), congbao.chinhphu.vn (Công báo).
  * *Thương mại (bổ sung metadata):* thuvienphapluat.vn, luatvietnam.vn (Tra cứu hiệu lực, quan hệ văn bản).
* **Bản quyền:** VBQPPL là public domain (Điều 15 khoản 2 Luật SHTT) — thu thập hợp pháp.
* **Bộ dữ liệu public (VN):**
  * *VLQA:* 3.129 câu hỏi | 59.636 điều
  * *ALQAC 2023:* 100 câu hỏi | 2.131 điều
  * *Zalo LTR 2021:* 3.196 câu hỏi | 61.425 điều
  * *TVPL:* 175.334 câu hỏi (165.334 train + 10.000 test) | 224.006 passage. (Zalo'21 sau làm sạch: 8.436 VB / 114.177 điều).
* **Tổng kho CSDL Quốc gia:** TW: 22 bộ/ngành | Địa phương: 34 tỉnh/thành.

### Cấu trúc & phân cấp văn bản (VD: Luật 04/2017/QH14)
* **Hệ thống phân cấp:** (Điều 63 NĐ 78/2025 — mỗi tầng "hoặc không có")
  * Phần (tùy chọn) – Chương – Mục (tùy chọn) – Tiểu mục (tùy chọn) – **Điều** ← neo – Khoản (tùy chọn) – Điểm (tùy chọn).
* **Viện dẫn chéo:** Khoản 3 Điều 8 → "Điều 9 của Luật này"; Khoản 2 Điều 6 → "điểm a,b,c khoản 1 Điều này".
* **Ví dụ — Luật 04/2017/QH14:**
  * Quốc hội · 12/6/2017 · 4 chương · 35 điều
  * Chương I: Quy định chung (Điều 1 phẳng, Điều 3 Giải thích từ ngữ, Điều 4 Tiêu chí DNNVV: khoản 1 -> điểm a, b)
  * Chương II: Nội dung hỗ trợ (Mục 1: Hỗ trợ chung -> Điều 8–11...)
  * *Lưu ý:* Không có Phần; Chương I không có Mục.

### Luồng xử lý & lưu ý – dữ liệu luật VN
* **Pipeline (3 bước):**
  1. *Thu thập:* Crawl vbpl.vn, lưu bản gốc + metadata.
  2. *Tách mức Điều:* Dựng cây cấu trúc, tách tới từng Điều.
  3. *Chunk + Index:* Nhỏ theo cấu trúc (khoản/điểm).
* **Lưu ý xử lý (đặc thù dữ liệu luật VN):**
  * Sửa đổi liên tục (có VB sửa nhiều lần).
  * Hợp nhất bắt buộc (Đ.168 Luật 2015), phủ không đều → ưu tiên bản hợp nhất.
  * Hiệu lực một phần gắn trạng thái ở mức điều/khoản, không chỉ mức văn bản.
  * Quan hệ nhiều tầng Luật – NĐ – TT; "1 VB sửa nhiều VB" (NĐ 34/2016).
  * Chồng chéo, mâu thuẫn → giải quyết xung đột: ưu tiên thứ bậc + thời điểm.
  * Chất lượng nguồn → đối soát vbpl ↔ TVPL.

### Bộ dữ liệu SME Law QA – Tổng quan & Độ khó
* **Quy mô:** 2.000 câu hỏi.
  * Độ dài TB: 37,8 từ (166 ký tự); trung vị 37.
  * Ngắn nhất 10 từ · Dài nhất 89 từ.
* **Phân phối độ dài câu hỏi:** 10–14 (2%), 15–19 (12%), 20–29 (29%), 30–49 (28%), 50+ (29%).
* **Phân loại câu hỏi theo dạng:**
  * Thủ tục/ Hồ sơ: 23,8%
  * Điều kiện/ Yêu cầu: 20,2%
  * Khác: 17,6%
  * Quyền & Nghĩa vụ: 13,7%
  * Tình huống giả định: 8,3%
  * Thời hạn: 6,2%
  * Xử phạt/ Vi phạm: 5,9%
  * Hỗ trợ/ Ưu đãi: 3,9%
  * So sánh/ Khác biệt: 0,3%
* **Độ khó:** ~57% câu ≥ 30 từ · 38,4% cross-doc (≥2 VB) · 8,3% scenario · 515 câu multi-hop.
* **Thách thức chính:** Multi-hop retrieval.

### Case Studies
#### 1. Dễ (ID 1577)
* **Tình huống:** 10 từ · 1 văn bản · 1 điều. *"Pháp luật quy định những phương tiện quảng cáo nào?"*
* **Văn bản & điều:** Luật Quảng cáo 16/2012/QH13 — Điều 17 (liệt kê các phương tiện quảng cáo).
* **Năng lực kiểm tra:** Tra cứu định nghĩa/liệt kê thẳng. 1 nguồn · 1 điều — không cần phân rã, không multi-hop. Baseline: RAG cơ bản cũng trả lời đúng.

#### 2. Trung bình (ID 2)
* **Tình huống:** 15 từ · 2 văn bản (cross-doc). *"Doanh nghiệp nhỏ và vừa được hưởng ưu đãi gì khi tham gia đấu thầu?"*
* **Phân rã → điều luật:**
  * Ưu đãi dành cho DNNVV (chính sách hỗ trợ) → Luật Hỗ trợ DNNVV 04/2017, Điều 13.
  * Cơ chế ưu đãi trong đấu thầu → Luật Đấu thầu 22/2023, Điều 10.
* **Năng lực kiểm tra:** Cross-doc: pull đúng 2 chunks từ 2 luật chuyên ngành. Tổng hợp ưu đãi từ 2 nguồn, không trùng/sót. Mỗi nguồn 1 điều rõ ràng — chưa multi-hop sâu.

#### 3. Khó (ID 1128)
* **Tình huống:** 69 từ · 2 văn bản · 10 điều. *"...đối thủ sao chép trái phép phần mềm để cho thuê thu lợi và làm mất khách hàng; cần xác định hành vi xâm phạm quyền tác giả ở điểm nào, cách tính tổn thất về cơ hội kinh doanh ra sao, và phải chuẩn bị tài liệu/ chứng cứ gì khi gửi đơn yêu cầu xử lý?"*
* **Phân rã 3 yêu cầu → điều luật:**
  1. *Hành vi xâm phạm ở điểm nào?* Luật SHTT Đ.20, 22 (quyền tác giả với phần mềm) · Đ.28 (hành vi xâm phạm).
  2. *Tính tổn thất/ cơ hội kinh doanh?* Luật SHTT Đ.204 (nguyên tắc xác định thiệt hại) · Đ.205 (căn cứ mức bồi thường).
  3. *Tài liệu, chứng cứ khi gửi đơn?* Luật SHTT Đ.198 + NĐ 17/2023 Đ.66, 73, 75, 76 (trình tự, hồ sơ xử lý).
* **Năng lực kiểm tra:** Phân rã truy vấn: 3 mệnh đề pháp lý độc lập. Multi-hop: gom đúng 10 điều từ 2 văn bản. Cross-doc reasoning: nối quyền (Luật SHTT) ↔ chế tài/thủ tục (NĐ 17/2023). Tổng hợp mạch lạc cả 3 phần.

#### 4. Tình huống thực tế (ID 1720)
* **Tình huống:** 88 từ · 3 văn bản · 4 điều. *"...lô hàng nhập khẩu nghi xâm phạm quyền tác giả; muốn yêu cầu hải quan kiểm soát, đồng thời thuê giám định viên xác minh — vậy nghĩa vụ bảo đảm tài chính ra sao, hợp đồng giám định cần nội dung chính nào, và nếu đối tượng bị xâm phạm là người tiêu dùng dễ bị tổn thương thì công ty có trách nhiệm gì?"*
* **Phân rã 4 mệnh đề con:**
  1. Yêu cầu hải quan kiểm soát hàng nhập khẩu nghi xâm phạm.
  2. Nghĩa vụ bảo đảm tài chính khi yêu cầu kiểm soát/giám định.
  3. Nội dung chính của hợp đồng giám định.
  4. Trách nhiệm khi đối tượng là người tiêu dùng dễ bị tổn thương.
* **Văn bản:** 3 văn bản · 4 điều (SHTT · kiểm soát hải quan · bảo vệ NTD).
* **Năng lực kiểm tra:** Phân rã 4 vế đan xen nhiều lĩnh vực. Gom 3 văn bản khác lĩnh vực; nối SHTT + hải quan + bảo vệ NTD. Thực tiễn cao — tiêu biểu câu hỏi SME thực tế sẽ đặt ra.

---

## 3. Các phương pháp đánh giá

### Đánh giá Statute Retrieval
* **Precision@k (P@k):** $P@k = \frac{|Relevant \cap Topk|}{k}$
  * P@1: Điều luật đầu tiên có đúng không?
  * P@5: Tỉ lệ đúng trong top 5 kết quả.
* **Recall@k (R@k):** $R@k = \frac{|Relevant \cap Topk|}{|Relevant|}$
  * Hệ thống có tìm thấy tất cả các điều luật liên quan trong top k kết quả trả về không?
* **Fβ Score:** $F_\beta = (1 + \beta^2) \cdot \frac{P@k \cdot R@k}{(\beta^2 \cdot P@k) + R@k}$
  * β = 1 (F1): Cân bằng Precision và Recall.
  * β = 2 (F2): Ưu tiên Recall, đánh giá Recall quan trọng gấp đôi Precision.
* **MRR (Mean Reciprocal Rank):** $MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}$
  * Đánh giá vị trí của điều luật đúng đầu tiên. Rank càng nhỏ (vị trí càng cao), MRR càng lớn.

#### Ví dụ đánh giá Statute Retrieval
* **Kịch bản:** Truy vấn về "Thủ tục đăng ký tạm trú".
  * Ground Truth (R): Điều 27, Điều 28 (Tổng R = 2)
  * Kết quả trả về (Top 5): [Điều 5, Điều 27, Điều 12, Điều 28, Điều 9]
* **Tính toán:**
  1. **Precision@5:** P = 2/5 = 0.4 (40%)
  2. **Recall:** R = 2/2 = 1.0 (100%)
  3. **MRR:** Kết quả đúng đầu tiên ở Rank 2 → RR = 1/2 = 0.5
  4. **F1 Score:** $F1 = 2 \cdot \frac{0.4 \cdot 1.0}{0.4 + 1.0} \approx 0.57$

### Đánh giá Statute QA
* **Accuracy (Yes/No):** $\frac{\text{\# đúng}}{\text{\# tổng}}$
* **Truyền thống (Lexical):**
  * ROUGE-L/BLEU: Độ trùng lặp từ vựng.
  * Exact Match (EM): Trùng khớp 100%.
  * BERTScore: Đánh giá độ tương đồng ngữ nghĩa.
* **LLM-as-a-Judge & Human Eval:**
  * Khắc phục hạn chế của độ đo truyền thống về mặt ngữ nghĩa pháp lý.
  * *Correctness:* Đúng bản chất pháp lý không?
  * *Hallucination Rate:* Sinh ra luật "ảo"?
  * *Faithfulness:* Trích dẫn đúng điều/khoản.
  * *Fluency:* Mức độ lưu loát, trôi chảy.

#### Ví dụ: Prompt cho LLM-as-a-Judge
*Chấm điểm câu trả lời (Dựa trên hướng dẫn của Chuyên Gia)*
* **System Prompt:** Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.
* **User Prompt (Trích lược):**
  * Return format:
    1. Explanation: Briefly explain your reasoning...
    2. Constructive feedback: Provide neutral, constructive feedback...
    3. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0... strictly follow the format: `[[score]]`.
  * Warnings:
    * (+) means affirmed, (-) means denied, (-/+) indicates both acceptable.
    * Deviations should be penalized unless legally correct.
    * Statutes should be cited precisely (Abs., Ziff., lit.).
  * Inputs: `Question`, `Reference Answer`, `Model's Answer`.

#### Ví dụ đánh giá Statute QA
* **Câu hỏi:** Trẻ em dưới 14 tuổi có được cấp Căn cước không?
* **Đáp án:** Có. Theo Luật Căn cước 2023, công dân VN dưới 14 tuổi được cấp thẻ căn cước theo nhu cầu.
* **Mô hình A (Trùng lặp từ vựng cao - Đúng bản chất):**
  * *Trả lời:* Theo Luật Căn cước 2023, công dân VN dưới 14 tuổi được cấp thẻ.
  * *Đánh giá:* ROUGE/BLEU: Cao. LLM/Human Eval: Tốt (Chính xác, đủ ý cơ bản).

### Nguyên tắc đánh giá trong Hệ thống hỏi đáp pháp lý
| Giai đoạn | Metric | Câu hỏi kiểm tra |
|---|---|---|
| **1. Retrieval** | Recall, F2-score, MAP | Đã trích xuất đủ và đúng điều luật/tiền lệ chưa? |
| **2. QA** | Accuracy, ROUGE | Lập luận pháp lý có chính xác và hợp lệ không? |
| **3. End-to-End** | Cascading Errors | Garbage in, garbage out. Truy xuất sai ⇒ Phán quyết sai. |

---

## 4. Tổng thể kiến trúc Retrieval và Question Answering

### Kiến trúc tổng thể: End-to-End System
* **Luồng:** Câu hỏi → Embedding → Retrieval → Re-rank → QA → Phản hồi
* **Chức năng:** Hiểu ý định → Lọc sơ bộ → Lọc kỹ → Trả lời
* **Nguyên tắc làm việc:** Đầu vào (Retrieval) quyết định Đầu ra (QA). Tìm sai luật ⇒ Dễ trả lời sai.

### Từng thành phần: Retrieval (Phương pháp cơ bản)
1. **Tra từ khóa chính xác (BM25):**
   * Giống như tra mục lục ở cuối sách.
   * Hoạt động cực nhanh, không cần huấn luyện.
   * Đòi hỏi từ ngữ phải trùng khớp chính xác.
   * Yếu khi dùng từ đồng nghĩa (ví dụ: "ly hôn" vs "chấm dứt hôn nhân").
2. **Tra theo nghĩa khái niệm (Dense):**
   * Xếp tài liệu theo "ngữ nghĩa".
   * Mô hình chuyển câu hỏi và luật thành vector ngữ nghĩa.
   * Hiểu các từ diễn đạt khác nhau cùng bản chất.
   * Không lo bị lỗi viết tắt hay diễn đạt vòng vo.

### Từng thành phần: Retrieval (Nâng cao & Thực tiễn)
3. **Tìm kiếm Kết hợp (Hybrid Retrieval):**
   * Kết hợp cả Từ khóa + Nghĩa khái niệm.
   * Thuật toán tự động trộn hai kết quả (RRF).
   * Tránh bỏ sót các thuật ngữ pháp lý đặc thù.
   * Thực tế: Cho kết quả toàn diện và chính xác nhất.
4. **Huấn luyện AI cho Tiếng Việt:**
   * Tinh chỉnh cho hệ thống pháp luật VN.
   * Dựa trên nền tảng mô hình tiếng Việt gốc.
   * Dạy AI nhận diện các cặp câu hỏi-đáp pháp lý thực tế (Fine-tuning).
   * Thích ứng riêng cho cấu trúc và văn phong văn bản luật VN.

### Từng thành phần: Lọc kỹ & Soạn thảo
* **Đối chiếu chi tiết (Re-ranking):**
  * Đặt câu hỏi và điều luật cạnh nhau, đối chiếu kỹ lưỡng từng từ ngữ.
  * Sắp xếp lại danh sách 100 điều luật từ thô đến tinh.
  * Độ chính xác cực cao.
  * Tốn năng lực xử lý (chỉ chạy trên danh sách rút gọn).
  * *Luồng:* Câu hỏi + Điều luật → [Bộ lọc chi tiết] → Điểm phù hợp.
* **RAG (Retrieval-Augmented Generation):**
  * AI (LLM) trả lời dựa trên tài liệu được cung cấp sẵn.
  * AI làm bài thi dưới dạng "đề mở" (Open-book).
  * Không tự bịa luật (hạn chế tối đa ảo giác AI).
  * Tự động trích dẫn chính xác điều/khoản làm căn cứ.
  * *Luồng:* Đáp án = AI (Câu hỏi + Các điều luật đã chọn).

### Tại sao bước Tìm luật (Retrieval) phải đúng?
* **Tình huống thực tế:** Câu hỏi: "Điều kiện để doanh nghiệp nhỏ và vừa được hỗ trợ?"
* **✓ Tìm đúng điều luật:**
  * Tìm ra Điều 5 Luật Hỗ trợ DNNVV.
  * Tìm ra Điều 5 NĐ-80 quy định chi tiết.
  * *AI trả lời:* Đáp ứng đúng tiêu chí về lao động (≤ 200 người) và vốn (≤ 100 tỷ)...
  * *Kết quả:* Chính xác, tin cậy.
* **× Tìm sai điều luật:**
  * Tìm ra Điều 1 về Phạm vi áp dụng chung.
  * Tìm ra luật về Doanh nghiệp nhà nước.
  * *AI trả lời:* Giải thích mông lung hoặc bịa ra tiêu chí mới (Hallucination)...
  * *Kết quả:* Legally Wrong (Sai pháp lý).
* **Nguyên lý cốt lõi: Garbage In, Garbage Out.** Đầu tư vào hệ thống truy xuất đúng luật là yếu tố quan trọng của mọi kiến trúc AI.

---

## 5. Các bộ dữ liệu & Benchmark

### Tổng quan các bộ dữ liệu về Legal QA & Retrieval
*VLQA là một bộ tiêu chuẩn đánh giá tiếng Việt toàn diện, quy mô lớn và được chú thích bởi chuyên gia, bao trùm 27 lĩnh vực pháp lý.*

| Bộ dữ liệu | # Câu hỏi | # Căn cứ | Dạng câu trả lời | Lĩnh vực | Nguồn | Ngôn ngữ |
|---|---|---|---|---|---|---|
| COLIEE’24 | 1,206 | 768 | Đúng/Sai | Luật dân sự | Đề thi luật | ja, en |
| JEC-QA | 26,365 | 3382 | TN | Luật thành văn | Đề thi luật | zh |
| PrivacyQA | 1,750 | 335 | Đa trích xuất | Luật bảo mật | Luật gia | en |
| EQUALS | 6,914 | 3,081 | Dạng dài | Luật thành văn | Diễn đàn | zh |
| LLeQA | 1,868 | 27,942 | Dạng dài | Luật thành văn | Luật gia | fr |
| ALQAC’23 | 100 | 2,131 | Đúng/Sai, TN | Luật thành văn | Luật gia | vi |
| ViRHE4QA | 9,758 | 294 | Dạng dài | Giáo dục | Sinh viên | vi |
| **VLQA** | **3,129** | **59,636** | **Dạng dài** | **Luật thành văn** | **Diễn đàn** | **vi** |

### Bộ dữ liệu Quốc tế COLIEE: Tổng quan & Kết quả
*COLIEE (Competition on Legal Information Extraction and Entailment) là kỳ thi năng lực thuật toán thường niên uy tín toàn cầu trong cộng đồng Legal NLP, đồng tổ chức cùng hội nghị quốc tế ICAIL.*
* **Dữ liệu nguồn:** Sử dụng các bài thi tư pháp chính thức và Bộ luật Dân sự Nhật Bản (hỗ trợ song ngữ Anh-Nhật).

**Task 3: Statute Retrieval (Full Leaderboard)**
| Team | return | retr. | F2 | prec. | rec. |
|---|---|---|---|---|---|
| JNLP | 102 | 75 | 0.836 | 0.804 | 0.874 |
| CAPTAIN | 93 | 73 | 0.830 | 0.833 | 0.852 |
| INFA | 73 | 56 | 0.692 | 0.767 | 0.683 |
| AIIRLab | 219 | 78 | 0.667 | 0.356 | 0.886 |
| OVGU | 87 | 51 | 0.604 | 0.635 | 0.614 |
| UI | 80 | 46 | 0.582 | 0.586 | 0.589 |
| UA | 365 | 36 | 0.254 | 0.099 | 0.436 |

**Task 4: Statute QA (Formal Run)**
| Team | Lang. | Correct | Acc. |
|---|---|---|---|
| KIS | J | 66 | 0.904 |
| CAPTAIN | E | 60 | 0.822 |
| JNLP | E | 59 | 0.808 |
| UA | E | 57 | 0.781 |
| KLAP | E | 56 | 0.767 |
| OVGU | E | 54 | 0.740 |
| Baseline | - | 36 | 0.507 |

### Benchmark Tiếng Việt VLQA: Tổng quan & Hiệu năng Truy xuất
*VLQA là bộ tiêu chuẩn đánh giá tiếng Việt quy mô lớn dành riêng cho các tác vụ tìm kiếm và hỏi đáp pháp lý thành văn. Bao gồm 3,129 câu hỏi tự luận dạng dài đi kèm hệ thống 59,636 căn cứ pháp lý được gán nhãn chi tiết.*

**Kết quả thực nghiệm của hệ thống cải tiến đa tầng trên tập kiểm thử VLQA**
| Mô hình (Model) | F2 | Precision | Recall | MRR |
|---|---|---|---|---|
| **One-stage retriever (Truy xuất 1 tầng)** | | | | |
| BM25 | 0.3368 | 0.2265 | 0.3835 | 0.3780 |
| BGE-m3 | 0.4783 | 0.3222 | 0.5442 | 0.5654 |
| **Two-stage retriever (Truy xuất 2 tầng)** | | | | |
| mBERT-ft | 0.5515 | 0.3740 | 0.6257 | 0.6675 |
| **Three-stage retriever (Truy xuất 3 tầng)** | | | | |
| GPT-4o-mini | 0.6703 | 0.4319 | 0.7776 | 0.7914 |
| Qwen3-30B-A3B | 0.7283 | 0.6523 | 0.7501 | 0.8057 |

*Nhận xét:* Kiến trúc kết hợp tái xếp hạng đa tầng (Three-stage) với các LLM mạnh mẽ giúp cải thiện đáng kể độ phủ pháp lý so với các mô hình truyền thống.

### Benchmark Tiếng Việt VLQA: Hiệu năng Hỏi đáp (QA)
* **Phân tích tác vụ Sinh (Generation):**
  * Các LLMs vượt trội hoàn toàn so với mô hình trích xuất cả về độ khớp từ vựng lẫn ngữ nghĩa.
  * Tinh chỉnh (Fine-tuning - ft) giúp mô hình local đạt kết quả rất sát với các dòng thương mại.
* **Điểm sáng công nghệ:**
  * Qwen2.5-14B (ft) đạt điểm ROUGE-L (0.6606) cao nhất, chứng minh khả năng bám sát lập luận xuất sắc.
  * GPT-4o-mini (2-shot) dẫn đầu về BERTScore (0.8639), tối ưu nhất về hiểu và diễn đạt ngữ nghĩa pháp lý.

**Đánh giá hiệu năng trên tập VLQA**
| Mô hình | Cài đặt | ROUGE-L | BERTScore |
|---|---|---|---|
| **Mô hình trích xuất (Extractive)** | | | |
| PhoBERT | ft | 0.2683 | 0.7304 |
| **Mô hình sinh văn bản (Seq2Seq)** | | | |
| BARTpho | ft | 0.3318 | 0.7872 |
| **Mô hình ngôn ngữ lớn (LLMs)** | | | |
| Qwen2.5-14B | ft | 0.6606 | 0.8504 |
| GPT-4o | 2-shot | 0.6212 | 0.8533 |
| GPT-4o-mini | 2-shot | 0.6492 | 0.8639 |

---

## 6. Thách thức và Hướng tiếp cận

1. **Luật pháp thay đổi liên tục:**
   * *Vấn đề:* Mô hình huấn luyện năm 2020 có thể bị lỗi thời vào năm 2025 do các văn bản luật mới được ban hành.
   * *Tiếp cận:* Xây dựng hệ thống tự động cập nhật và re-index cơ sở dữ liệu liên tục (Dynamic RAG).
2. **Dữ liệu tiếng Việt còn hạn chế:**
   * *Vấn đề:* Sự thiếu hụt các bộ dữ liệu chất lượng cao để đánh giá độ tin cậy của thuật toán.
   * *Tiếp cận:* Cấp thiết xây dựng các bộ đánh giá Legal NLP ở cấp độ chuyên gia (Professional-level benchmark) cho Việt Nam.
3. **Hiện tượng Ảo giác (Hallucination):**
   * *Vấn đề:* LLM có xu hướng tự tin sinh ra câu trả lời nghe rất hợp lý nhưng sai lệch về mặt pháp lý.
   * *Tiếp cận:* Giới hạn không gian sinh văn bản thông qua Grounding và kiểm chứng chéo bằng logic rules.
4. **Khoảng cách liên ngành:**
   * *Vấn đề:* Luật sư không hiểu sâu về cơ chế hoạt động của AI; trong khi kỹ sư AI thiếu kiến thức pháp lý hàn lâm.
   * *Tiếp cận:* Triển khai quy trình Human-in-the-loop (HITL), để chuyên gia pháp lý trực tiếp hướng dẫn và gán nhãn.

---

## 7. Tổng kết: 5 Kết luận trọng tâm (Key Takeaways)

1. **Đặc thù hệ thống pháp luật:** Việt Nam áp dụng hệ thống luật thành văn (Statute Law). Nghiên cứu cần trọng tâm vào năng lực truy xuất và đối sánh văn bản quy phạm thay vì án lệ.
2. **Lĩnh vực rủi ro cao (High-stakes):** Sinh văn bản trôi chảy là chưa đủ. Sự chính xác, Sự tin cậy và Tính minh bạch là tiêu chuẩn tối thượng.
3. **Bài toán nút thắt (Bottleneck):** Tối ưu hóa truy xuất là tiên quyết nhằm ngăn ngừa lan truyền lỗi và nâng cao hiệu năng.
4. **Rào cản dữ liệu (Low-resource):** Dữ liệu huấn luyện tiếng Việt còn hạn chế. Kỹ thuật tạo sinh dữ liệu giả lập (Synthetic Data) và tự huấn luyện (Self-training) là chiến lược tất yếu.
5. **Hướng tới hệ sinh thái mở:** *Vietnamese Legal Benchmark Coming Soon!* Giải quyết bài toán Legal NLP đòi hỏi nỗ lực liên ngành. Kỳ vọng dữ liệu mở sẽ thúc đẩy hợp tác giữa cộng đồng AI & Chuyên gia pháp lý.

---

## Tài liệu tham khảo

1. Yu Fan et al. “LEXam: Benchmarking Legal Reasoning on 340 Law Exams”. In: *The Fourteenth International Conference on Learning Representations*. 2026. url: https://openreview.net/forum?id=xNhbMyXsJn.
2. Randy Goebel et al. “An Overview of the COLIEE 2025 Competition: Legal Case Law and Statute Law Information Retrieval and Entailment”. In: *Proceedings of the Twentieth International Conference on Artificial Intelligence and Law*. 2025, pp. 506–515.
3. Tan-Minh Nguyen et al. “Vlqa: The first comprehensive, large, and high-quality vietnamese dataset for legal question answering”. In: *arXiv preprint arXiv:2507.19995* (2025).

---
*Xin trân trọng cảm ơn!*  
*Thảo luận và Trả lời câu hỏi*  
*Statute Law Retrieval & Question Answering | Open dataset coming soon*