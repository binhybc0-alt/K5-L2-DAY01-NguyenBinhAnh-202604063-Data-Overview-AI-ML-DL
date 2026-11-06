## 1. Phân loại ảnh

1. Chép record hạng 1:

```text
class_id = 468
class_name = cab
rank = 1
score = 0.510915
taxonomy_name = ImageNet-1K
```

2. Record này nói về cả bức ảnh hay một vật thể riêng trong ảnh? Vì sao?

**Trả lời:**  
Record này mô tả cả bức ảnh. Đây là bài toán phân loại ảnh (image classification), nên mô hình chỉ đưa ra một nhãn cho toàn bộ ảnh thay vì từng vật thể riêng lẻ.

3. Danh sách tên lớp (class list) do đâu mà có?

**Trả lời:**  
Danh sách tên lớp không phải do mô hình tự tạo ra. Các lớp được lấy từ taxonomy ImageNet-1K đi kèm với checkpoint `yolo11n-cls.pt`.

4. Vì sao cần ghi cả class_id, class_name và taxonomy_name?

**Trả lời:**  
Cần ghi đầy đủ `class_id`, `class_name` và `taxonomy_name` để tránh nhầm lẫn giữa các bộ dữ liệu khác nhau. Mỗi taxonomy có thể dùng mã lớp khác nhau cho cùng một tên lớp.

5. Nếu trong ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?

**Trả lời:**  
Guideline cần quy định rõ cách chọn lớp chính, ví dụ chọn đối tượng nổi bật nhất hoặc chủ đề chiếm phần lớn nội dung ảnh để đảm bảo tính nhất quán khi gán nhãn.

6. Vì sao không được lấy score của model làm ground truth?

**Trả lời:**  
Score chỉ thể hiện mức độ tự tin của mô hình đối với prediction. Ground truth phải được tạo ra bởi con người dựa trên guideline nên không thể dùng score của model làm nhãn chuẩn.

---

## 2. Phát hiện vật thể

1. Chép một record:

```text
class_name = person
score = 0.912625
bbox_xyxy = [385.33, 69.24, 498.92, 348.92]
bbox_width = 113.58
bbox_height = 279.68
```

2. Giải thích box bằng lời thường:

Bounding box bắt đầu từ điểm `(385.33, 69.24)` ở góc trên bên trái và kết thúc tại điểm `(498.92, 348.92)` ở góc dưới bên phải. Hộp có chiều rộng khoảng 113.58 pixel và chiều cao khoảng 279.68 pixel. Bounding box này bao quanh một người trong ảnh `kitchen`.

3. Chọn hai ngưỡng (ví dụ 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

**Trả lời:**  
*(Vui lòng kiểm tra con số chính xác từ kết quả chạy notebook của bạn để điền vào đây)*

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), reviewer phải làm thêm việc gì?

**Trả lời:**  
Khi hạ ngưỡng confidence, số lượng prediction tăng lên. Reviewer cần kiểm tra và loại bỏ các prediction sai (false positives) hoặc có độ tin cậy thấp.

5. Viết một quy tắc box để người gán nhãn làm theo.

**Trả lời:**  
Bounding box phải bao phủ toàn bộ phần vật thể nhìn thấy được và bám sát biên ngoài của vật thể, hạn chế chứa vùng nền không liên quan.

6. Trên ảnh kitchen, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline hay đưa lên hỏi Lab Coach?

**Trả lời:**  
Trong ảnh `kitchen` có một số vật thể bị che khuất hoặc nằm sát mép ảnh. Tôi sẽ bổ sung quy tắc vào guideline rằng vẫn gán nhãn phần vật thể còn nhìn thấy nếu có thể nhận diện được. Nếu không xác định được lớp của vật thể thì sẽ escalation cho Lab Coach.

---

## 3. Phân đoạn theo từng đối tượng

1. Thông tin một record mẫu từ `sample_id = "kitchen"` (`kitchen-002`):

- **`instance_id`**: `kitchen-002`
- **`class_name`**: `bowl`
- **`score`**: `0.735743`
- **`polygon_point_count`**: `67`
- **`polygon_xy` (3–5 điểm đầu)**: `[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]`

2. So với bounding box, polygon cung cấp thêm thông tin gì về hình dạng vật thể?

- **Độ chính xác hình học chi tiết (Contour/Shape)**: Bounding box chỉ cung cấp một hình chữ nhật bao ngoài vùng chứa vật thể, dẫn đến việc chứa nhiều khoảng trống thuộc về nền (background). Trong khi đó, **polygon** vạch sát đường viền thực tế của vật thể, giúp nắm bắt hình dáng không đều, các góc cạnh, đường cong, hoặc phần lõm của đối tượng.

3. Ý nghĩa của `instance_id` (Không phải class ID và không phải tracking ID):

- **Định danh cá thể độc lập**: `instance_id` dùng để phân biệt từng thực thể riêng biệt trong cùng một khung hình (ví dụ: phân biệt rõ ràng cái bát thứ nhất `kitchen-002` với cái bát thứ hai `kitchen-003`).
- **Không phải class ID**: Class ID phân loại loại vật thể chung (ví dụ: tất cả các loại "bát" đều chung một class ID), còn instance ID phân biệt từng đối tượng cụ thể dù chúng cùng loại.
- **Không phải tracking ID**: Tracking ID dùng để liên kết và theo dõi một đối tượng xuyên suốt qua chuỗi nhiều khung hình (video) theo thời gian, trong khi instance ID ở đây chỉ định danh tĩnh cho đối tượng trong một ảnh đơn lẻ.

4. Quy tắc vẽ biên mask cho người gán nhãn:

> **Quy tắc vẽ biên**: "Mask phải ôm sát ranh giới thực tế của vật thể, đi khít theo đường biên tiếp xúc với nền hoặc các vật thể xung quanh. Tuyệt đối không để hở khoảng trống lớn hoặc lẹm sâu vào thân vật thể. Đối với các vùng bị che khuất một phần hoặc biên mờ, người gán nhãn cần dựa vào ngữ cảnh không gian và phần hiển thị rõ để phỏng đoán và hoàn thiện đường biên một cách hợp lý nhất."

5. Xử lý biên mờ / hai vật chạm nhau / bị che trên ảnh `kitchen`:

- **Xác định**: Trên ảnh `kitchen`, các vật thể như bát (`kitchen-002`, `kitchen-003`) đặt trên bàn hoặc người đứng gần khu vực bếp thường xảy ra hiện tượng **chạm nhau** hoặc **che khuất**.
- **Lựa chọn**: Sử dụng **guideline** (hướng dẫn gán nhãn có sẵn) trước.
- **Lý do**: Các trường hợp biên mờ, vật chạm nhau hay che khuất cơ bản đều đã được chuẩn hóa phương án xử lý trong tài liệu guideline của dự án (ví dụ: quy tắc ưu tiên vật nằm phía trước, cách tách biên tại điểm giao nhau). Chỉ khi tình huống quá phức tạp, mờ nhòe hoàn toàn không thể phân định bằng mắt hoặc mâu thuẫn với tài liệu hướng dẫn thì mới cần **escalation** lên cấp quản lý/chuyên gia.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Nhãn lớp duy nhất gán cho toàn bộ bức ảnh (ví dụ: `class_name = cab` trong taxonomy ImageNet-1K). | Ảnh chứa nhiều chủ đề/vật thể cùng lúc; model chọn nhãn phụ (`score = 0.510915`) chưa nêu bật ngữ cảnh chính. | Đọc guideline để xác định đối tượng nổi bật nhất/chủ đề chính và chọn đúng 1 nhãn lớp duy nhất. | Kiểm tra nhãn lớp được chọn có phản ánh đúng ngữ cảnh chính của ảnh và chuẩn theo taxonomy không. |
| **Phát hiện vật thể** | Tọa độ Bounding Box dạng `[x_min, y_min, x_max, y_max]` kèm nhãn lớp (ví dụ: `person`). | Ngưỡng thấp sinh ra nhiều hộp ảo; vật bị che khuất/cắt mép (`bbox_xyxy` của `person` bị khuất một phần). | Vẽ Bounding Box ôm sát ranh giới nhìn thấy của đối tượng, bỏ qua nền nhiễu. | Kiểm tra độ khít của Bounding Box, soát lỗi bỏ sót vật thể và loại bỏ các hộp dự đoán sai (false positive). |
| **Instance segmentation** | Tọa độ đa giác chi tiết (`polygon_xy`) kèm `instance_id` từng vật thể (ví dụ: `kitchen-002`, `bowl`). | Vật thể chạm/chồng lên nhau (hai cái bát `kitchen-002` và `kitchen-003`), đường viền mờ nhòe, khó tách ranh giới. | Vẽ đa giác ôm sát viền thực tế từng cá thể đối tượng; tra cứu guideline hoặc báo escalation nếu biên mờ. | Kiểm tra độ chi tiết/mịn của viền đa giác, mã `instance_id` tách biệt và cách xử lý điểm giao nhau giữa các vật. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu):**  
   Tuyệt đối không đưa họ tên, mã số sinh viên, thông tin định danh cá nhân (PII) hoặc dữ liệu nhạy cảm vào trong tệp báo cáo cũng như repository public.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**  
   Dừng công việc ngay lập tức và báo cáo cho Giảng viên hướng dẫn, Trợ giảng (TA) hoặc Quản lý dự án (Project Manager).

---

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