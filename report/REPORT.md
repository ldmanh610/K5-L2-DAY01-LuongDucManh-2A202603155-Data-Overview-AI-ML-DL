# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:*11/6/2026*

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
> Bản ghi này mô tả cho toàn bộ bức ảnh chứ không phải một vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
> Danh sách này đi kèm với taxonomy/checkpoint (cụ thể ở đây là tập nhãn chuẩn của ImageNet-1K gồm 1000 lớp mà mô hình YOLOv11-cls đã được huấn luyện trên đó). Mô hình không thể tự nghĩ ra các tên lớp mới nằm ngoài tập dữ liệu huấn luyện này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
> Tránh trùng lặp & nhầm lẫn và co tính nhất quán.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
> Quy tắc đối tượng chính (Dominant Object)
> Quy tắc bối cảnh tổng thể (Global Context)
> Quy tắc phân cấp ưu tiên
6. Vì sao không được lấy `score` của model làm ground truth?
  score: Điểm số độ tin cậy của thuật toán toán học tại thời điểm dự đoán, nó không đại diện cho thực tế khách quan còn Ground truth phải là sự thật tuyệt đối.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= bus; `score`= 0.912557 ; `bbox_xyxy`=[93.17,187.95,223.01,320.91] ; `bbox_width`= 129.84 ; `bbox_height`= 132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
> Đối tượng được phát hiện là xe buýt (bus).
> Góc trên-trái của box nằm tại tọa độ (93.17, 187.95) px, tức cách mép trái ảnh khoảng 93 px và cách mép trên ảnh khoảng 188 px.
> Góc dưới-phải của box nằm tại tọa độ (223.01, 320.91) px, tức cách mép trái ảnh khoảng 223 px và cách mép trên ảnh khoảng 321 px.
> Box có:
> Chiều rộng: 129.84 px
> Chiều cao: 132.96 px
> Nói đơn giản: hộp bao phủ khu vực chứa xe buýt, bắt đầu từ vùng gần giữa bên trái của ảnh và kéo dài khoảng 130 × 133 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
> Với dữ liệu vd: kết quả là 1 và 1.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
> reviewer phải xem và xác nhận nhiều box hơn, đặc biệt là các box chất lượng thấp dễ bị sai
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
> Vẽ bounding box ôm sát phần nhìn thấy của đối tượng, để mép box tiếp xúc gần nhất với biên ngoài của vật thể và không bao gồm quá nhiều nền xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
> Đưa vào guideline nếu: Vẫn gán nhãn đối tượng nếu có thể nhận dạng chắc chắn, và box chỉ bao quanh phần nhìn thấy của vật thể."
> Escalation lên Lab Coach nếu:Không rõ vật thể thuộc lớp nào.Mức che khuất quá lớn khiến không thể xác định chắc chắn.Có nhiều cách diễn giải hợp lý dẫn đến nhãn khác nhau.
> Kết luận: Với các trường hợp che khuất/cắt mép thông thường nên đưa vào guideline để mọi annotator làm giống nhau; chỉ escalate khi không thể xác định nhãn hoặc quy tắc chưa đủ rõ để ra quyết định nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= traffic-001 ; `class_name`= 5 ; `score`= 0.925745 ; số điểm polygon=`polygon_point_count`= 120 ; 3–5 điểm đầu của `polygon_xy`= [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
