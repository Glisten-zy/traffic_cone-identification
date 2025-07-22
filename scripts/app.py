import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk
import requests
import base64
from io import BytesIO

API_URL = "http://127.0.0.1:8000/predict"

def upload_and_send():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    with open(path, "rb") as f:
        files = {"file": (path, f, "image/jpeg")}
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
        except Exception as e:
            text_box.config(state=tk.NORMAL)
            text_box.delete("1.0", tk.END)
            text_box.insert(tk.END, f"❌ 请求失败: {e}")
            text_box.config(state=tk.DISABLED)
            return

    data = response.json()
    caption1 = data.get("caption_template", "N/A")
    caption2 = data.get("caption_blip", "N/A")
    vis_base64 = data.get("vis_image_base64", "")

    if vis_base64.startswith("data:image"):
        vis_base64 = vis_base64.split(",", 1)[1]
    img_data = base64.b64decode(vis_base64)
    image = Image.open(BytesIO(img_data)).resize((480, 360))

    tk_img = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img

    text_box.config(state=tk.NORMAL)
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, f"📄 Template Description:\n{caption1}\n\n🧠 BLIP Output:\n{caption2}")
    text_box.config(state=tk.DISABLED)

# ===== Tkinter UI 布局 =====
root = tk.Tk()
root.title("🚦 Traffic Cone Captioning Client")

# 美化主题字体
default_font = ("Helvetica", 11)

# 设置主窗口大小并居中
root.geometry("600x600")
root.configure(bg="#f5f5f5")

# 图像显示区域
canvas_frame = tk.LabelFrame(root, text="📸 图像预览", font=default_font, bg="white", padx=5, pady=5)
canvas_frame.pack(pady=10)
canvas = tk.Canvas(canvas_frame, width=480, height=360, bg="#ddd")
canvas.pack()

# 上传按钮
btn = tk.Button(root, text="📂 上传图像", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", command=upload_and_send)
btn.pack(pady=10, ipadx=10, ipady=4)

# 文本框结果显示
text_frame = tk.LabelFrame(root, text="📋 描述结果", font=default_font, padx=5, pady=5)
text_frame.pack(fill="both", expand=True, padx=10, pady=10)
text_box = Text(text_frame, height=10, font=default_font, wrap=tk.WORD)
text_box.pack(fill="both", expand=True)
text_box.config(state=tk.DISABLED)

root.mainloop()
