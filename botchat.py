# botchat.py
import json
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mo_hinh_best.h5')
JSON_PATH = os.path.join(os.path.dirname(__file__), 'dieu_tri.json')


print("Đang load model, chờ chút...")
mo_hinh = load_model(MODEL_PATH)
print("Load model xong.")

try:
    INPUT_SHAPE = tuple(mo_hinh.input_shape[1:3])
except Exception:
    INPUT_SHAPE = (320, 320)  

cac_lop = [
    'binh_thuong',
    'lua_bi_nam',
    'nhan_khong_benh',
    'sau_cuon_la',
    'than_thu'
]



with open(JSON_PATH, "r", encoding="utf-8") as f:
    thong_tin_benh = json.load(f)

NGUONG = 0.6

def du_doan_benh(duong_dan_anh, nguong=NGUONG):
    """
    Trả về dict:
    {
      "success": True/False,
      "confidence": 0.xx,
      "label": "sau_cuon_la",
      "text": "chuỗi mô tả (dùng để hiển thị)",
      "info": { ... }  # dict từ JSON nếu có
    }
    """
    
    img = image.load_img(duong_dan_anh, target_size=INPUT_SHAPE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = mo_hinh.predict(arr)
    confidence = float(np.max(preds))
    idx = int(np.argmax(preds[0]))
    label = cac_lop[idx]

    
    if label == 'unknown' or confidence < nguong:
        return {
            "success": False,
            "confidence": round(confidence, 4),
            "label": None,
            "text": f"Không xác định được (Độ tin cậy: {confidence:.2%}). Hãy thử ảnh khác hoặc chụp rõ hơn.",
            "info": None
        }

    
    info = thong_tin_benh.get(label, None)
    if info:
        
        text_lines = []
        text_lines.append(f" Dự đoán: {label} (Độ tin cậy: {confidence:.2%})")
        ng = info.get("nguyen_nhan")
        if ng: text_lines.append(f" Nguyên nhân: {ng}")
        dh = info.get("dau_hieu")
        if dh: text_lines.append(f" Dấu hiệu: {', '.join(dh)}")
        tt = info.get("ten_thuoc")
        if tt: text_lines.append(f" Thuốc: {', '.join(tt)}")
        ll = info.get("lieu_luong")
        if ll: text_lines.append(f" Liều lượng: {ll}")
        hd = info.get("huong_dan_phun")
        if hd: text_lines.append(f" Hướng dẫn: {hd}")

        text = "\n".join(text_lines)
        return {
            "success": True,
            "confidence": round(confidence, 4),
            "label": label,
            "text": text,
            "info": info
        }
    else:
        return {
            "success": True,
            "confidence": round(confidence, 4),
            "label": label,
            "text": f" Dự đoán: {label} (Độ tin cậy: {confidence:.2%})\n⚠ Không có dữ liệu tư vấn trong JSON.",
            "info": None
        }
