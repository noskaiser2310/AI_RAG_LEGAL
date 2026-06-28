# R2AI2026 BUILD AI LEGAL ASSISTANT

> Trợ lý Pháp lý AI cho Doanh nghiệp SME
> Nền tảng: [leaderboard.aiguru.com.vn](https://leaderboard.aiguru.com.vn/competitions/13/)

---

## 1. Tổng quan

Cuộc thi về **Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt (Vietnamese Legal Information Retrieval & Question Answering)** do AI Guru – Dagoras Group tổ chức.

- **Số người tham gia:** 72
- **Số bài nộp:** 514 (tính đến hiện tại)
- **Trạng thái:** Đang diễn ra (phase công khai)

---

## 2. Bài toán

Xây dựng hệ thống AI giải quyết 2 nhiệm vụ:

### 2.1. Truy hồi thông tin pháp luật (IR)

Với tập câu hỏi Q = {q₁, q₂, ..., qₙ} và kho điều luật A = {a₁, a₂, ..., aₙ}, xác định tập con A′ ⊂ A gồm các điều luật "liên quan" đến câu hỏi. Một điều luật được coi là liên quan nếu câu hỏi có thể trả lời Có/Không dựa trên điều luật đó.

### 2.2. Hỏi đáp pháp luật (QA)

Dựa trên các điều luật đã truy hồi, sinh câu trả lời cho câu hỏi pháp lý. Hệ thống phải hiểu và suy luận nội dung pháp lý.

---

## 3. Mục tiêu

| # | Mục tiêu | Mô tả |
|---|----------|-------|
| 1 | Tra cứu pháp lý chính xác | Tra cứu điều khoản Luật Doanh nghiệp và văn bản liên quan SME |
| 2 | Hỏi đáp pháp lý tiếng Việt | Hiểu ngôn ngữ tự nhiên tiếng Việt, hỏi đáp tình huống pháp lý |
| 3 | Dẫn nguồn điều luật | Trích dẫn điều/khoản/văn bản, hiển thị nguồn tham chiếu |
| 4 | Tư vấn sơ bộ & cảnh báo | Hướng dẫn pháp lý sơ bộ, nhắc rủi ro tuân thủ, cảnh báo giới hạn AI |
| 5 | Kiểm soát nội dung sai lệch | Hạn chế hallucination, tránh bịa điều luật, tăng độ tin cậy |

---

## 4. Timeline

| Mốc | Ngày | Ghi chú |
|-----|------|---------|
| Khai mạc & phát hành test set | **03/06/2026** | |
| Đóng cổng nộp bài | **30/06/2026** | 23:59 UTC+7 |
| Công bố Top 10 → DemoDay | **05/07/2026** | |
| DemoDay & kết quả chung cuộc | **11/07/2026** | |

---

## 5. Dữ liệu

### 5.1. Đầu vào (test set)

BTCC cung cấp duy nhất test set. **Không có train/dev set.**

```json
{
  "id": <integer>,
  "question": "<string>"
}
```

### 5.2. Nguồn dữ liệu tự thu thập

Các đội tự chủ động thu thập:
- Văn bản pháp luật, thông tư, nghị định từ nguồn chính thống
- Dữ liệu SME (thuế, lao động, hợp đồng,...)
- Open dataset Legal NLP
- Mọi nguồn hợp pháp khác

### 5.3. Định dạng nộp bài

File `results.json` → nén vào `submission.zip` (phẳng, không thư mục con).

```json
[
  {
    "id": <integer>,
    "question": "<string>",
    "answer": "<string>",
    "relevant_docs": ["<mã VB>|<tên VB>"],
    "relevant_articles": ["<mã VB>|<tên VB>|<điều>"]
  }
]
```

**Quy tắc đặt tên văn bản:** `Loại văn bản + Mã văn bản + Trích yếu`

Ví dụ: `04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa`

---

## 6. Evaluation Metrics

### 6.1. Information Retrieval

Đánh giá tự động, trích xuất pattern `Điều X` từ trường `answer` / `relevant_docs` / `relevant_articles`, so sánh với đáp án chuẩn.

| Metric | Công thức |
|--------|-----------|
| **Precision** | macro avg(số điều đúng / số điều đã truy hồi) |
| **Recall** | macro avg(số điều đúng / số điều liên quan) |
| **F2-Macro** | (5 × P × R) / (4 × P + R) |

### 6.2. Question Answering

LLM-as-a-Judge + Chuyên gia pháp luật đánh giá 5 tiêu chí:

| # | Tiêu chí | Loại |
|---|----------|------|
| 1 | Căn cứ chính xác pháp luật | Tự động |
| 2 | Chính xác nội dung | Thủ công |
| 3 | Đầy đủ & toàn diện | Thủ công |
| 4 | Thực tiễn & áp dụng | Thủ công |
| 5 | Rõ ràng & dễ hiểu | Thủ công |

> ⚠️ 4 chỉ số thủ công hiện đang ở giá trị 0.0, sẽ được cập nhật sau khi giám khảo đánh giá.

### 6.3. Leaderboard columns

| Cột | Key | Sort |
|-----|-----|------|
| ARTICLES F2-MACRO | `ARTICLES_F2MACRO` | desc |
| DOCS F2-MACRO | `DOCS_F2MACRO` | desc |
| ARTICLES PRECISION | `ARTICLES_PRECISION` | desc |
| ARTICLES RECALL | `ARTICLES_RECALL` | desc |
| DOCS PRECISION | `DOCS_PRECISION` | desc |
| DOCS RECALL | `DOCS_RECALL` | desc |
| CHÍNH XÁC NỘI DUNG | `CHINH_XAC_NOI_DUNG` | desc |
| ĐẦY ĐỦ & TOÀN DIỆN | `DAY_DU` | desc |
| THỰC TIỄN & ÁP DỤNG | `THUC_TIEN` | desc |
| RÕ RÀNG & DỄ HIỂU | `RO_RANG` | desc |

---

## 7. Quy định mô hình

| Tiêu chí | Yêu cầu |
|----------|---------|
| Kích thước | **< 14B** tham số |
| Thời điểm | Phát hành **trước 01/03/2026** |
| Giấy phép | **Open-source**, tải trọng số tự do |
| Cấm | Mô hình đóng (GPT-4o, Gemini, Claude,...) |

---

## 8. Quy định nộp bài

| Phase | Thời gian | Max/ngày | Max tổng |
|-------|-----------|----------|----------|
| **Kiểm thử công khai** (Public) | 31/05 → 30/06/2026 | 10 | 210 |
| **Kiểm thử riêng** (Private) | 01/07 → 12/07/2026 | 5 | **5** |

Lưu ý:
- Private phase chỉ được **5 bài tổng cộng** — chọn lọc kỹ
- Đánh giá QA **không tự động**: đội phải **promote** bài nộp lên leaderboard
- Đánh giá QA định kỳ **mỗi tuần 1 lần**
- Kết quả chưa chính thức cho đến khi nộp working notes paper

---

## 9. Ban Tổ chức

**AI Guru – Công ty CP Tập đoàn Dagoras Group**
- Địa chỉ: Tầng 8, 80 Duy Tân, Cầu Giấy, Hà Nội
- Nguyễn Thị Minh Nguyệt — 0981544974
- Vũ Thị Thuỳ Linh — 0961891198

---

## 10. Điều khoản & Quy định chung

- BTCC có quyền hủy bỏ, sửa đổi hoặc loại tư cách
- Khi nộp bài, đồng ý công bố điểm số công khai
- Mỗi người **1 tài khoản duy nhất**
- Được phép lập đội, **không tham gia nhiều đội**
- 1 tài khoản/đội được phê duyệt nộp bài
- Nghiêm cấm gian lận, lừa dối, chơi không công bằng
