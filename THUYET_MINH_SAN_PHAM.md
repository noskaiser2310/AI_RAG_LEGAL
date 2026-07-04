# TÀI LIỆU THUYẾT MINH SẢN PHẨM

**Tên dự án/Sản phẩm:** Hệ thống Hỏi đáp Pháp luật Việt Nam (Vietnamese Legal RAG)
**Cuộc thi:** ROAD TO AI (R2AI) 2026
**Đội thi:** [Điền tên đội của bạn]

---

## 1. Tài liệu mô tả dữ liệu

Hệ thống RAG Pháp luật Việt Nam được xây dựng dựa trên tập hợp 6 nguồn dữ liệu chính, được xử lý và chuẩn hóa sâu (semantic chunking, metadata enrichment) để tối ưu hóa khả năng tìm kiếm (retrieval) và sinh ngôn ngữ (generation).

### a. Nguồn dữ liệu sử dụng
| # | Tên nguồn | Số lượng | Đặc tả nội dung | Mức độ ưu tiên |
|---|-----------|----------|-----------------|----------------|
| 1 | `tmquan/phapdien-moj-gov-vn` | 202k điều luật | Dữ liệu pháp điển hóa từ moj.gov.vn | Cao nhất (Primary) |
| 2 | `th1nhng0/vietnamese-legal-documents` | 153k văn bản | Metadata văn bản pháp luật, hiệu lực | Bổ trợ (Supplement) |
| 3 | `undertheseanlp/UTS_VLC` | 318 bộ luật | Văn bản luật toàn vẹn định dạng Markdown | Bổ trợ (Supplement) |
| 4 | `tmquan/pbgdpl-vn-legal-qna` | 4.5k Q&A | Các tình huống pháp lý thực tế có trích dẫn | Kiểm chứng (Validation) |
| 5 | `duyet/vietnamese-legal-instruct` | ~100k Q&A | Tập dữ liệu tinh chỉnh Legal Instruction | Bổ trợ (Supplement) |
| 6 | `tmquan/anle-toaan-gov-vn` | ~1k án lệ | Án lệ thực tế từ Tòa án nhân dân Tối cao | Ngữ cảnh (Context) |

### b. Cấu trúc dữ liệu và Định dạng
* **Dữ liệu thô (Raw Chunks):** Được phân tách theo ngữ nghĩa pháp luật (Điều, Khoản), giữ nguyên Metadata (ID điều luật, tiêu đề chương, trạng thái hiệu lực) và lưu dưới định dạng **JSONL**.
* **Dữ liệu chỉ mục (Indexes):**
  * *Dense Index:* Lưu trữ dưới định dạng `.index` bằng thư viện FAISS (Vector nhúng 768-D).
  * *Sparse Index:* Lưu trữ model BM25 (dạng `.pkl` hoặc thư mục `bm25s`).

### c. Hướng dẫn truy cập và sử dụng
* **Link Google Drive chia sẻ dữ liệu chỉ mục (Indexes):** 
  [https://drive.google.com/drive/u/1/folders/1FSFKamm_fwss0sI4wBnBOxOko1gIGMH9](https://drive.google.com/drive/u/1/folders/1FSFKamm_fwss0sI4wBnBOxOko1gIGMH9)
* **Cách sử dụng:** Trong mã nguồn, đội thi đã chuẩn bị sẵn script để tự động tải và giải nén toàn bộ kho dữ liệu này vào đúng thư mục làm việc. Khảo khảo phần lệnh cài đặt ở mục 4.

---

## 2. Mô hình sử dụng

Hệ thống kết hợp ba loại mô hình để tạo thành một pipeline RAG (Retrieve-then-Read) hoàn chỉnh. Tất cả đều là mã nguồn mở và có kích thước tối ưu (<14B tham số), đáp ứng đúng tiêu chí của cuộc thi.

### a. Thông tin về mô hình và Phiên bản Checkpoint
1. **Mô hình Sinh ngôn ngữ (LLM - Generator):**
   * **Checkpoint:** `Qwen/Qwen2.5-7B-Instruct` (hoặc `Qwen3-8B-Instruct`)
   * **Định dạng:** Chạy Offline qua HF Transformers / vLLM. Khuyến nghị Quantization 4-bit (bitsandbytes) để tối ưu VRAM (~6GB).
2. **Mô hình Nhúng (Embedding Model):**
   * **Checkpoint:** `mainguyen9/vietlegal-harrier-0.6b` (Qwen3Model Decoder)
   * **Cấu hình:** Sử dụng Last-token pooling.
3. **Mô hình Xếp hạng lại (Reranker Model):**
   * **Checkpoint:** `AITeamVN/Vietnamese_Reranker`
   * **Cấu hình:** Cross-Encoder score calibration.

### b. Hướng dẫn tải, sử dụng & Link truy cập Checkpoint
Tất cả các checkpoints đều được công khai (Public) trên nền tảng HuggingFace Hub. Quá trình tải về được tự động hóa hoàn toàn thông qua thư viện `transformers` và `sentence-transformers` khi hệ thống khởi chạy lần đầu.

* **Link Checkpoint LLM:** [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* **Link Checkpoint Embedding:** [https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b)
* **Link Checkpoint Reranker:** [https://huggingface.co/AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)

---

## 3. Mã nguồn

### a. Toàn bộ mã nguồn
Toàn bộ mã nguồn của hệ thống được đính kèm trong thư mục nộp bài hoặc có thể truy cập thông qua kho chứa Github công khai (nếu có).
* **Link Github (Nội bộ):** [Dán link repo Github của bạn vào đây]

### b. Danh sách thư viện và Dependencies
Danh sách các thư viện cần thiết được khai báo đầy đủ trong tệp `requirements.txt`. Các thư viện cốt lõi bao gồm:
* Khung RAG và DL: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `vllm`
* Tìm kiếm: `faiss-cpu`, `rank-bm25`, `bm25s`
* Tiện ích: `numpy`, `scikit-learn`, `datasets`, `tqdm`, `pydantic-settings`, `gdown`

### c. Tệp cấu hình cần thiết
* **`.env` (hoặc `.env.example`):** Tệp lưu trữ biến môi trường (ví dụ: `DEVICE=cuda`).
* Các siêu tham số (hyperparameters) phục vụ retrieval và generation được quản lý tại module `src/retrieval/adaptive.py` và `src/generator/generator.py`.

---

## 4. Tài liệu hướng dẫn (Tái hiện sản phẩm)

Hướng dẫn chi tiết từng bước để Giám khảo / Hội đồng có thể dễ dàng chạy lại sản phẩm từ đầu. Thông tin này cũng được đối chiếu trong tệp `README.md` đi kèm.

### a. Yêu cầu môi trường
* **Hệ điều hành:** Linux (Ubuntu) hoặc Windows (WSL2).
* **RAM Hệ thống:** 16GB+ (Khuyến nghị 32GB+ để nạp toàn bộ Corpus nếu cần).
* **GPU (VRAM):** Tối thiểu 8GB VRAM (để chạy model Qwen 8B 4-bit) và 24GB VRAM nếu không dùng lượng tử hóa.
* **Môi trường Cloud:** Tương thích hoàn toàn với Kaggle Notebook (GPU T4x2).

### b. Các bước cài đặt và vận hành

**Bước 1: Thiết lập môi trường và cài đặt thư viện**
```bash
# Tạo môi trường ảo (khuyến nghị dùng Conda hoặc venv)
conda create -n legalrag python=3.10 -y
conda activate legalrag

# Cài đặt các phụ thuộc
pip install -r requirements.txt
```

**Bước 2: Tải dữ liệu Index (từ Google Drive)**
Chạy lệnh sau để tải và tự động giải nén dữ liệu FAISS/BM25 vào thư mục `data/indexes/`:
```bash
python scripts/download_index_gdrive.py
```

**Bước 3: Thiết lập cấu hình**
Tạo file `.env` ở thư mục gốc với nội dung:
```env
DEVICE=cuda
HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
# Các tham số khác (nếu có) xem trong .env.example
```

**Bước 4: Thực thi hệ thống (Inference)**
Để chạy hệ thống đánh giá ngoại tuyến (Offline inference) nhằm cho ra kết quả dự đoán (answers):
```bash
python scripts/submit.py
```
Hệ thống sẽ:
1. Tự động tải các checkpoint (LLM, Embedding, Reranker) từ HuggingFace (nếu chưa có).
2. Load bộ dữ liệu FAISS/BM25 đã tải ở Bước 2.
3. Chạy qua tập dữ liệu test và xuất kết quả F2-Macro tại `results/submission.json`.

*(Ghi chú: Đội thi đã đính kèm một file chạy mẫu trên Kaggle tại `notebooks/kaggle_submission_qwen3.ipynb` minh họa cách chạy trực tiếp hệ thống trên môi trường T4x2)*.
