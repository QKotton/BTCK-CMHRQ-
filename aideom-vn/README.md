# 🇻🇳 AIDEOM-VN — Mô hình ra quyết định: Kinh tế Việt Nam trong kỉ nguyên AI

Web app **Streamlit** tương tác giải trọn bộ **12 bài tập** của đề "Bộ bài tập thực hành Mô hình
ra quyết định — Phát triển kinh tế Việt Nam trong kỉ nguyên AI" (UEB — Viện QTKD), dùng dữ liệu
Việt Nam 2020–2025.

**Tính năng cốt lõi:** mọi bảng số liệu (macro, ngành, vùng, hệ số β, danh mục dự án, kịch bản…)
đều **chỉnh sửa trực tiếp trên giao diện** — lời giải tối ưu, biểu đồ và phân tích chính sách.

| Bài | Nội dung | Kỹ thuật |
|---|---|---|
| 1 | Cobb-Douglas mở rộng, TFP, phân rã tăng trưởng, dự báo 2030 | numpy/pandas |
| 2 | LP 4 hạng mục + shadow price + độ nhạy Z*(B) | scipy HiGHS + PuLP |
| 3 | Chỉ số Priority 10 ngành + heatmap độ nhạy a₆ | min-max, trọng số |
| 4 | LP 24 biến vùng×hạng mục + chi phí của công bằng (C5) | PuLP/CBC |
| 5 | MIP chọn 15 dự án (loại trừ, precedence, đa năm, E[Z]) | PuLP/CBC |
| 6 | TOPSIS 6 vùng + trọng số Entropy + độ nhạy w_AI | numpy |
| 7 | NSGA-II 4 mục tiêu, biên Pareto 3D, nghiệm thỏa hiệp TOPSIS | pymoo |
| 8 | Tối ưu động 2026-2035, cú sốc 2028, front-load vs trải đều | scipy SLSQP |
| 9 | NetJob lao động, ngưỡng đào tạo lại, Sankey | scipy HiGHS |
| 10 | Stochastic 2 giai đoạn: VSS, EVPI, minimax regret | PuLP/CBC |
| 11 | Q-learning 81 trạng thái × 5 hành động, so với rule-based | numpy (API kiểu gymnasium) |
| 12 | Dashboard AIDEOM-VN 6 module, 5 kịch bản S1–S5 | tích hợp Bài 1-11 |



NSO/GSO, Bộ KH&CN (MoST), WIPO GII 2025, World Bank — số liệu trong CSV đã làm tròn phục vụ
giảng dạy theo lưu ý sư phạm của đề; khi viết báo cáo cần truy xuất số gốc từ nso.gov.vn.


