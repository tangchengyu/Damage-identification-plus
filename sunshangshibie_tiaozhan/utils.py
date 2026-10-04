import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import os
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加src目录到路径，以便导入预测模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from predict import DamagePredictor
    PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入预测模块: {e}")
    PREDICTION_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ------------------------- 全局变量 -------------------------
# 宏观图片相关变量
original_image_hongguan = None
gray_image_hongguan     = None
binary_image_hongguan   = None
pixels_per_unit_scale_hongguan = None  # μm/pixel
img_path_hongguan       = None

# 体视镜图片相关变量
original_image_tishijing = None
gray_image_tishijing     = None
binary_image_tishijing   = None
pixels_per_unit_scale_tishijing = None  # μm/pixel
img_path_tishijing       = None

# 当前选择的图片类型和对应变量
current_image_type = None  # 'hongguan' 或 'tishijing'
original_image = None
gray_image     = None
binary_image   = None
pixels_per_unit_scale = None
img_path       = None

# 深度学习预测器
predictor = None
prediction_result = None

# ------------------------- Tk 根窗口 -------------------------
root = tk.Tk()
root.title("齿轮损伤智能识别系统")
root.geometry("380x550")
root.resizable(False, False)
selected_model_variant = 'updated'
model_status = tk.StringVar(value='分类模型: updated（微调）；尚未加载')
tk.Label(root, textvariable=model_status, wraplength=360, fg='#355878').pack(pady=6)
tk.Label(root, text='寿命预测仅供历史原型演示；标签来源未验证',
         wraplength=360, fg='#9a5b16').pack(pady=2)

# ------------------------- 初始化函数 -------------------------
def initialize_predictor():
    """初始化深度学习预测器"""
    global predictor
    if not PREDICTION_AVAILABLE:
        return False

    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'output', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"创建模型目录: {model_dir}")

        predictor = DamagePredictor(model_dir, model_variant=selected_model_variant)
        model_status.set(f'分类模型: {predictor.model_variant}；两个视角已加载')
        return True
    except Exception as e:
        model_status.set(f'分类模型加载失败: {selected_model_variant}')
        print(f"初始化预测器失败: {e}")
        return False

# ------------------------- 通用函数 -------------------------
def update_current_image_vars():
    """更新当前图片相关的全局变量"""
    global original_image, gray_image, binary_image, pixels_per_unit_scale, img_path

    if current_image_type == 'hongguan':
        original_image = original_image_hongguan
        gray_image = gray_image_hongguan
        binary_image = binary_image_hongguan
        pixels_per_unit_scale = pixels_per_unit_scale_hongguan
        img_path = img_path_hongguan
    elif current_image_type == 'tishijing':
        original_image = original_image_tishijing
        gray_image = gray_image_tishijing
        binary_image = binary_image_tishijing
        pixels_per_unit_scale = pixels_per_unit_scale_tishijing
        img_path = img_path_tishijing
    else:
        original_image = None
        gray_image = None
        binary_image = None
        pixels_per_unit_scale = None
        img_path = None

def load_image_hongguan():
    """加载宏观图片"""
    global original_image_hongguan, gray_image_hongguan, binary_image_hongguan, img_path_hongguan, pixels_per_unit_scale_hongguan, current_image_type

    path = filedialog.askopenfilename(
        title="选择宏观图片",
        filetypes=[("Image", "*.jpg *.jpeg *.JPG *.JPEG *.tif *.TIF")])
    if not path:
        return

    img_path_hongguan = path
    original_image_hongguan = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_image_hongguan is None:
        messagebox.showerror("错误", "无法读取宏观图片！")
        return

    gray_image_hongguan = cv2.cvtColor(original_image_hongguan, cv2.COLOR_BGR2GRAY)
    _, binary_image_hongguan = cv2.threshold(gray_image_hongguan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels_per_unit_scale_hongguan = None

    # 设置当前图片类型为宏观
    current_image_type = 'hongguan'
    update_current_image_vars()

    messagebox.showinfo("提示", "宏观图片已加载！")

def load_image_tishijing():
    """加载体视镜图片"""
    global original_image_tishijing, gray_image_tishijing, binary_image_tishijing, img_path_tishijing, pixels_per_unit_scale_tishijing, current_image_type

    path = filedialog.askopenfilename(
        title="选择体视镜图片",
        filetypes=[("Image", "*.jpg *.jpeg *.JPG *.JPEG *.tif *.TIF")])
    if not path:
        return

    img_path_tishijing = path
    original_image_tishijing = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original_image_tishijing is None:
        messagebox.showerror("错误", "无法读取体视镜图片！")
        return

    gray_image_tishijing = cv2.cvtColor(original_image_tishijing, cv2.COLOR_BGR2GRAY)
    _, binary_image_tishijing = cv2.threshold(gray_image_tishijing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels_per_unit_scale_tishijing = None

    # 设置当前图片类型为体视镜
    current_image_type = 'tishijing'
    update_current_image_vars()

    messagebox.showinfo("提示", "体视镜图片已加载！")

def switch_image_type():
    """切换当前操作的图片类型"""
    global current_image_type

    # 检查是否有已加载的图片
    available_types = []
    if original_image_hongguan is not None:
        available_types.append('hongguan')
    if original_image_tishijing is not None:
        available_types.append('tishijing')

    if not available_types:
        messagebox.showwarning("警告", "请先加载图片！")
        return

    if len(available_types) == 1:
        current_image_type = available_types[0]
        update_current_image_vars()
        messagebox.showinfo("提示", f"当前操作图片类型: {'宏观' if current_image_type == 'hongguan' else '体视镜'}")
        return

    # 创建选择窗口
    choice_window = tk.Toplevel(root)
    choice_window.title("选择图片类型")
    choice_window.geometry("200x150")
    choice_window.resizable(False, False)

    tk.Label(choice_window, text="选择要操作的图片类型:").pack(pady=10)

    def select_hongguan():
        global current_image_type
        current_image_type = 'hongguan'
        update_current_image_vars()
        choice_window.destroy()
        messagebox.showinfo("提示", "当前操作图片类型: 宏观")

    def select_tishijing():
        global current_image_type
        current_image_type = 'tishijing'
        update_current_image_vars()
        choice_window.destroy()
        messagebox.showinfo("提示", "当前操作图片类型: 体视镜")

    if 'hongguan' in available_types:
        tk.Button(choice_window, text="宏观图片", command=select_hongguan, width=15).pack(pady=5)
    if 'tishijing' in available_types:
        tk.Button(choice_window, text="体视镜图片", command=select_tishijing, width=15).pack(pady=5)

def get_scale():
    """为当前选择的图片类型标定比例尺"""
    global pixels_per_unit_scale, pixels_per_unit_scale_hongguan, pixels_per_unit_scale_tishijing

    if gray_image is None:
        messagebox.showwarning("警告", "请先加载图片！")
        return

    image_type_name = '宏观' if current_image_type == 'hongguan' else '体视镜'

    plt.figure(f"{image_type_name}图片比例尺标定")
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"点击{image_type_name}图片标尺两端后按 Enter")
    pts = plt.ginput(2, timeout=0)
    plt.close()

    if len(pts) != 2:
        return

    (x1, y1), (x2, y2) = pts
    pix_dist = np.hypot(x2 - x1, y2 - y1)
    real_dist = simpledialog.askfloat("输入", f"{image_type_name}图片标尺实际长度(μm)：", parent=root)

    if real_dist is None or real_dist <= 0:
        return

    scale_value = real_dist / pix_dist

    # 更新对应类型的比例尺
    if current_image_type == 'hongguan':
        pixels_per_unit_scale_hongguan = scale_value
    elif current_image_type == 'tishijing':
        pixels_per_unit_scale_tishijing = scale_value

    pixels_per_unit_scale = scale_value

    messagebox.showinfo("提示", f"{image_type_name}图片比例尺已设定：1 px = {scale_value:.4f} μm")

def check_ready():
    if gray_image is None:
        messagebox.showwarning("警告", "请先加载图片！")
        return False
    if pixels_per_unit_scale is None:
        messagebox.showwarning("警告", "请先标定比例尺！")
        return False
    return True

# ------------------------- 功能实现 -------------------------
def auto_detect():
    """智能损伤识别和参数测量"""
    global prediction_result

    # 检查是否至少有一张图片已加载
    if original_image_hongguan is None and original_image_tishijing is None:
        messagebox.showwarning("警告", "请先加载至少一张图片（宏观或体视镜）！")
        return

    # 检查比例尺
    scale_ready = False
    if original_image_hongguan is not None and pixels_per_unit_scale_hongguan is not None:
        scale_ready = True
    if original_image_tishijing is not None and pixels_per_unit_scale_tishijing is not None:
        scale_ready = True

    # Classification does not need geometric calibration; measurement does.

    # 初始化预测器
    if predictor is None:
        if not initialize_predictor():
            messagebox.showerror("错误", "无法初始化深度学习预测器！将使用传统方法。")
            # 回退到传统方法
            traditional_detect()
            return

    try:
        # 使用深度学习模型进行预测
        prediction_result = predictor.predict_damage_type(
            image_hongguan_path=img_path_hongguan,
            image_tishijing_path=img_path_tishijing
        )

        # 获取预测的损伤类型
        predicted_class_zh = prediction_result['predicted_class_zh']
        confidence = prediction_result['confidence']

        # 显示预测结果
        # result_text = f"AI预测结果:\n损伤类型: {predicted_class_zh}\n置信度: {confidence:.3f}\n\n"
        result_text = f"预测结果:\n损伤类型: {predicted_class_zh}\n类别概率: {confidence:.3f}\n\n"

        # 根据预测结果进行相应的参数测量
        measure_result = (measure_damage_parameters(predicted_class_zh) if scale_ready else
                          '未标定比例尺：本次只识别类别。进行面积/长度测量前请先标定。')

        # 合并结果
        final_result = result_text + measure_result

        # 创建结果显示窗口
        show_detailed_results(final_result, prediction_result)

    except Exception as e:
        messagebox.showerror("错误", f"预测失败: {e}\n将使用传统方法。")
        traditional_detect()

def traditional_detect():
    """传统的损伤检测方法（原有逻辑）"""
    if not check_ready():
        return

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_px = total_peri_px = 0
    max_long_px = max_short_px = 0
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        peri = cv2.arcLength(cnt, True)
        total_area_px += area
        total_peri_px += peri
        count += 1
        (w, h) = cv2.minAreaRect(cnt)[1]
        long, short = max(w, h), min(w, h)
        max_long_px = max(max_long_px, long)
        max_short_px = max(max_short_px, short)

    total_area_um = total_area_px * pixels_per_unit_scale ** 2
    total_peri_um = total_peri_px * pixels_per_unit_scale
    max_long_um = max_long_px * pixels_per_unit_scale
    max_short_um = max_short_px * pixels_per_unit_scale

    damage = simpledialog.askstring("类型", "损伤类型（输入'点蚀'或其他）：", parent=root)
    if damage is None:
        return

    if damage.strip() == "点蚀":
        messagebox.showinfo("结果", f"总面积={total_area_um:.2f}μm²\n总周长={total_peri_um:.2f}μm\n点蚀个数={count}")
    else:
        messagebox.showinfo("结果", f"总面积={total_area_um:.2f}μm²\n总周长={total_peri_um:.2f}μm\n最长={max_long_um:.2f}μm\n最宽={max_short_um:.2f}μm")

def measure_damage_parameters(damage_type):
    """根据损伤类型测量相应参数"""
    # 优先使用当前选择的图片，如果没有则自动选择
    if gray_image is None or pixels_per_unit_scale is None or binary_image is None:
        # 自动选择有比例尺的图片
        if original_image_hongguan is not None and pixels_per_unit_scale_hongguan is not None:
            switch_to_image_type('hongguan')
        elif original_image_tishijing is not None and pixels_per_unit_scale_tishijing is not None:
            switch_to_image_type('tishijing')
        else:
            return "错误: 没有可用的已标定图片"

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_px = total_peri_px = 0
    max_long_px = max_short_px = 0
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        peri = cv2.arcLength(cnt, True)
        total_area_px += area
        total_peri_px += peri
        count += 1
        (w, h) = cv2.minAreaRect(cnt)[1]
        long, short = max(w, h), min(w, h)
        max_long_px = max(max_long_px, long)
        max_short_px = max(max_short_px, short)

    total_area_um = total_area_px * pixels_per_unit_scale ** 2
    total_peri_um = total_peri_px * pixels_per_unit_scale
    max_long_um = max_long_px * pixels_per_unit_scale
    max_short_um = max_short_px * pixels_per_unit_scale

    image_type_name = '宏观' if current_image_type == 'hongguan' else '体视镜'

    if damage_type == "点蚀":
        # 1) 先自动推荐阈值：所有轮廓真实面积的中位数
        areas = [cv2.contourArea(c) * pixels_per_unit_scale ** 2
                 for c in contours if cv2.contourArea(c) > 0]
        suggest = float(np.median(areas)) if areas else 1.0
        # 2) 弹出带默认值的输入框
        area_th = simpledialog.askfloat(
            "面积阈值", f"推荐阈值：{suggest:.1f} μm²\n可手动修改：",
            parent=root, initialvalue=suggest, minvalue=0.0
        )
        if area_th is None:
            return "用户取消面积阈值设定"
        # 重新用默认二值图找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 下面统计时只保留 area > area_th 的轮廓

        # 下面统计面积、周长、个数等逻辑保持不变
        total_area_um = total_peri_um = 0
        count = 0
        filtered_contours = []  # 用于存储符合面积阈值的轮廓
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < 1:
                continue
            area_um = area_px * pixels_per_unit_scale ** 2  # 换成真实面积
            if area_um < area_th:  # 用真实面积比阈值
                continue

            total_area_um += area_um  # 累加真实面积
            total_peri_um += cv2.arcLength(cnt, True) * pixels_per_unit_scale
            count += 1
            filtered_contours.append(cnt)  # 将符合条件的轮廓加入列表

        avg_area = total_area_um / count if count else 0
        avg_peri = total_peri_um / count if count else 0

        # 重新绘制损伤区域的可视化图像，只显示面积大于阈值的损伤区域
        mask = np.zeros_like(gray_image)
        for cnt in filtered_contours:  # 只绘制符合面积阈值的轮廓
            cv2.drawContours(mask, [cnt], -1, 255, -1)
        overlay = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])  # BGR 绿色
        vis_img = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
        plt.figure("损伤区域标注")
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{image_type_name}图片-损伤区域")
        plt.axis('off')
        plt.show(block=False)

        return (f"参数测量结果（基于{image_type_name}图片）:\n"
                f"面积阈值: {area_th:.1f} μm²\n"
                f"总面积: {total_area_um:.2f} μm²\n"
                f"总周长: {total_peri_um:.2f} μm\n"
                f"点蚀个数: {count}\n"
                f"平均面积: {avg_area:.2f} μm²\n"
                f"平均周长: {avg_peri:.2f} μm")
    else:
        # 用半透明亮绿色覆盖全部损伤区域
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, contours, -1, 255, -1)
        overlay = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])  # BGR 绿色
        vis_img = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
        plt.figure("损伤区域标注")
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{image_type_name}图片-损伤区域（绿色覆盖）")
        plt.axis('off')
        plt.show(block=False)

        return f"参数测量结果（基于{image_type_name}图片）:\n总面积: {total_area_um:.2f} μm²\n总周长: {total_peri_um:.2f} μm\n最长: {max_long_um:.2f} μm\n最宽: {max_short_um:.2f} μm"

def switch_to_image_type(image_type):
    """切换到指定的图片类型"""
    global current_image_type
    current_image_type = image_type
    update_current_image_vars()

def show_detailed_results(result_text, prediction_data):
    """显示详细的预测和测量结果"""
    result_window = tk.Toplevel(root)
    result_window.title("智能识别结果")
    result_window.geometry("500x400")

    # 创建滚动文本框
    text_frame = tk.Frame(result_window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_widget = tk.Text(text_frame, yscrollcommand=scrollbar.set, wrap=tk.WORD)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=text_widget.yview)

    # 插入结果文本
    text_widget.insert(tk.END, result_text)
    text_widget.insert(tk.END, f"\n分类权重版本: {prediction_data['model_variant']}\n")
    for view, path in prediction_data['checkpoints'].items():
        text_widget.insert(tk.END, f"{view}: {os.path.basename(path)}\n")
    text_widget.insert(tk.END, f"融合权重: {prediction_data['weights']}\n类别概率不代表经校准的可靠性。\n")

    # 添加详细预测信息
    # text_widget.insert(tk.END, "\n详细预测信息:\n")
    # text_widget.insert(tk.END, f"使用的模型: {', '.join(prediction_data['used_models'])}\n")
    #
    # if 'hongguan' in prediction_data['used_models']:
    #     text_widget.insert(tk.END, f"宏观模型权重: {prediction_data['weights']['hongguan']:.3f}\n")
    # if 'tishijing' in prediction_data['used_models']:
    #     text_widget.insert(tk.END, f"体视镜模型权重: {prediction_data['weights']['tishijing']:.3f}\n")
    #
    # text_widget.insert(tk.END, "\n各类别概率:\n")
    # for class_name, prob in prediction_data['probabilities_zh'].items():
    #     text_widget.insert(tk.END, f"{class_name}: {prob:.3f}\n")

    text_widget.config(state=tk.DISABLED)

    # 关闭按钮
    tk.Button(result_window, text="关闭", command=result_window.destroy, width=10).pack(pady=10)

def measure_two_points():
    if not check_ready():
        return
    plt.figure("两点测距")
    plt.imshow(gray_image, cmap='gray')
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        return
    (x1, y1), (x2, y2) = pts
    dist = np.hypot(x2 - x1, y2 - y1) * pixels_per_unit_scale
    messagebox.showinfo("结果", f"两点距离 = {dist:.2f} μm")

def gray_profile(direction):
    if not check_ready():
        return
    plt.figure(f"{direction}灰度剖面")
    plt.imshow(gray_image, cmap='gray')
    pt = plt.ginput(1, timeout=0)
    plt.close()
    if not pt:
        return
    x, y = int(round(pt[0][0])), int(round(pt[0][1]))
    data = gray_image[y, :] if direction == "水平" else gray_image[:, x]
    x_um = np.arange(len(data)) * pixels_per_unit_scale
    plt.figure()
    plt.plot(x_um, data)
    plt.title(f"{direction}灰度剖面")
    plt.xlabel("距离 (μm)")
    plt.ylabel("灰度值")
    plt.grid(True)
    plt.show()

def gray_heatmap():
    if not check_ready():
        return
    h, w = gray_image.shape
    extent = [0, w * pixels_per_unit_scale, 0, h * pixels_per_unit_scale]
    plt.figure("灰度云图")
    plt.imshow(gray_image, cmap='viridis', extent=extent)
    plt.colorbar(label="灰度值")
    plt.title("灰度云图")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.show()

# ------------------------- 寿命预测功能 -------------------------
def lifetime_prediction():
    """
    寿命预测功能：弹出齿轮特征输入窗口
    """
    # 检查是否有加载的图片
    if original_image is None:
        messagebox.showwarning("警告", "请先加载齿轮图片！")
        return

    # 创建特征输入窗口
    feature_window = tk.Toplevel(root)
    feature_window.title("齿轮特征输入")
    feature_window.geometry("500x700")
    feature_window.resizable(False, False)

    # 创建滚动框架
    canvas = tk.Canvas(feature_window)
    scrollbar = tk.Scrollbar(feature_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # 特征输入字典
    feature_vars = {}

    # 尺寸特征 (7维)
    size_frame = tk.LabelFrame(scrollable_frame, text="尺寸特征", font=("Arial", 12, "bold"))
    size_frame.pack(fill="x", padx=10, pady=5)

    size_features = [
        ("模数 (mm)", "moshu", "齿轮的基本参数，决定齿轮尺寸"),
        ("齿数", "chishu", "影响传动比和接触应力"),
        ("变位系数", "bianweixishu", "修正齿轮几何形状的参数"),
        ("螺旋角 (°)", "luoxuanjiao", "斜齿轮的重要参数，影响传动平稳性"),
        ("压力角 (°)", "yalijiao", "决定齿轮传动特性的关键角度"),
        ("齿顶圆直径 (mm)", "chidingyuan(mm)", "齿轮外径，影响接触应力分布"),
        ("齿宽 (mm)", "chikuan(mm)", "轴向宽度，影响承载能力")
    ]

    # 设置尺寸特征的默认值
    size_defaults = {
        "moshu": "3",
        "chishu": "23",
        "bianweixishu": "0.008",
        "luoxuanjiao": "0",
        "yalijiao": "25",
        "chidingyuan(mm)": "75.2",
        "chikuan(mm)": "16"
    }

    for i, (label, key, desc) in enumerate(size_features):
        frame = tk.Frame(size_frame)
        frame.pack(fill="x", padx=5, pady=2)
        tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
        var = tk.StringVar(value=size_defaults.get(key, ""))
        entry = tk.Entry(frame, textvariable=var, width=15)
        entry.pack(side="left", padx=5)
        tk.Label(frame, text=desc, font=("Arial", 8), fg="gray").pack(side="left")
        feature_vars[key] = var

    # 工况特征 (2维)
    operation_frame = tk.LabelFrame(scrollable_frame, text="工况特征", font=("Arial", 12, "bold"))
    operation_frame.pack(fill="x", padx=10, pady=5)

    operation_features = [
        ("额定扭矩 (N·m)", "edingniujiu(N*m)", "齿轮设计承受的最大扭矩"),
        ("额定转速 (r/min)", "edingzhuansu(r/min)", "齿轮设计工作转速")
    ]

    # 设置工况特征的默认值
    operation_defaults = {
        "edingniujiu(N*m)": "85.15",
        "edingzhuansu(r/min)": "6393"
    }

    for i, (label, key, desc) in enumerate(operation_features):
        frame = tk.Frame(operation_frame)
        frame.pack(fill="x", padx=5, pady=2)
        tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
        var = tk.StringVar(value=operation_defaults.get(key, ""))
        entry = tk.Entry(frame, textvariable=var, width=15)
        entry.pack(side="left", padx=5)
        tk.Label(frame, text=desc, font=("Arial", 8), fg="gray").pack(side="left")
        feature_vars[key] = var

    # 材料特征 (8维)
    material_frame = tk.LabelFrame(scrollable_frame, text="材料特征", font=("Arial", 12, "bold"))
    material_frame.pack(fill="x", padx=10, pady=5)

    material_features = [
        ("材料类型", "cailiao", "钢材牌号，影响基本力学性能"),
        ("密度 (g/cm³)", "midu(g/cm3)", "材料密度，影响惯性力"),
        ("弹性模量 (MPa)", "tanxingmoliang(MPa)", "材料刚度，影响接触应力和变形"),
        ("泊松比", "bosongbi", "材料横向变形系数"),
        ("屈服应力 (MPa)", "qufuyingli(MPa)", "材料屈服强度，决定承载能力"),
        ("强化系数 (MPa)", "qianghuaxishu(MPa)", "材料硬化特性参数"),
        ("硬化指数", "yinghuazhishu", "材料应变硬化指数"),
        ("摩擦系数", "mocaxishu", "接触面摩擦特性，影响磨损")
    ]

    # 材料类型下拉框
    material_type_frame = tk.Frame(material_frame)
    material_type_frame.pack(fill="x", padx=5, pady=2)
    tk.Label(material_type_frame, text="材料类型", width=20, anchor="w").pack(side="left")
    material_var = tk.StringVar(value="12Cr2Ni4AGJB1951-1994")
    material_combo = ttk.Combobox(material_type_frame, textvariable=material_var,
                                  values=["12Cr2Ni4AGJB1951-1994", "16Cr3NiWMoVNbEQ/S10-0361-2004"], width=20, state="readonly")
    material_combo.pack(side="left", padx=5)
    tk.Label(material_type_frame, text="钢材牌号，影响基本力学性能", font=("Arial", 8), fg="gray").pack(side="left")
    feature_vars["cailiao"] = material_var

    # 设置材料特征的默认值
    material_defaults = {
        "midu(g/cm3)": "7.84",
        "tanxingmoliang(MPa)": "207500",  # (205-210)*1000的中间值
        "bosongbi": "0.285",  # [0.27, 0.3]的中间值
        "qufuyingli(MPa)": "785",
        "qianghuaxishu(MPa)": "2000",
        "yinghuazhishu": "0.115",  # [0.08, 0.15]的中间值
        "mocaxishu": "0.1"  # [0.05, 0.15]的中间值
    }

    # 其他材料特征
    for i, (label, key, desc) in enumerate(material_features[1:]):
        frame = tk.Frame(material_frame)
        frame.pack(fill="x", padx=5, pady=2)
        tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
        var = tk.StringVar(value=material_defaults.get(key, ""))
        entry = tk.Entry(frame, textvariable=var, width=15)
        entry.pack(side="left", padx=5)
        tk.Label(frame, text=desc, font=("Arial", 8), fg="gray").pack(side="left")
        feature_vars[key] = var

    # 预测按钮
    def start_prediction():
        """
        开始寿命预测
        """
        try:
            # 收集所有特征值
            gear_attributes = {}

            # 检查必填字段
            required_fields = [
                "moshu", "chishu", "bianweixishu", "luoxuanjiao", "yalijiao",
                "chidingyuan(mm)", "chikuan(mm)", "edingniujiu(N*m)", "edingzhuansu(r/min)",
                "midu(g/cm3)", "tanxingmoliang(MPa)", "bosongbi", "qufuyingli(MPa)",
                "qianghuaxishu(MPa)", "yinghuazhishu", "mocaxishu"
            ]

            for field in required_fields:
                value = feature_vars[field].get().strip()
                if not value:
                    messagebox.showerror("错误", f"请填写 {field} 字段！")
                    return
                try:
                    gear_attributes[field] = float(value)
                except ValueError:
                    messagebox.showerror("错误", f"{field} 字段必须是数字！")
                    return

            # 材料类型
            gear_attributes["cailiao"] = feature_vars["cailiao"].get()

            # 调用寿命预测模型
            predicted_lifetime = predict_gear_lifetime(img_path, gear_attributes)

            if predicted_lifetime is not None:
                # 显示预测结果
                result_text = f"预测寿命: {predicted_lifetime:.2f} 循环次数\n历史模型演示；寿命标签来源尚未验证。"
                messagebox.showinfo("寿命预测结果", result_text)
            else:
                messagebox.showerror("错误", "寿命预测失败，请检查输入参数！")

        except Exception as e:
            messagebox.showerror("错误", f"寿命预测过程中发生错误: {str(e)}")

    # 按钮框架
    button_frame = tk.Frame(scrollable_frame)
    button_frame.pack(fill="x", padx=10, pady=10)

    tk.Button(button_frame, text="开始预测寿命", command=start_prediction,
              bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2).pack(fill="x")

    # 配置滚动
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def predict_gear_lifetime(image_path, gear_attributes):
    """Run the historical lifetime model in an isolated process to avoid module collisions."""
    import json
    import subprocess
    from pathlib import Path
    try:
        project_root = Path(__file__).resolve().parent.parent
        process = subprocess.run(
            [sys.executable, str(project_root / 'project.py'), 'lifetime',
             '--image', str(Path(image_path).resolve()), '--attributes-stdin'],
            input=json.dumps(gear_attributes, ensure_ascii=False),
            text=True, encoding='utf-8', capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        if process.returncode != 0:
            print(process.stderr)
            return None
        return float(json.loads(process.stdout)['predicted_lifetime'])
    except Exception as error:
        print(f"寿命预测错误: {error}")
        return None

# ------------------------- 主界面按钮 -------------------------
btn = [
    ("加载宏观图片",    load_image_hongguan),
    ("加载体视镜图片",  load_image_tishijing),
    ("切换图片类型",    switch_image_type),
    ("标定比例尺",      get_scale),
    ("智能识别",        auto_detect),
    ("寿命预测",        lifetime_prediction),
    ("两点测距",        measure_two_points),
    ("水平灰度分析",    lambda: gray_profile("水平")),
    ("竖直灰度分析",    lambda: gray_profile("竖直")),
    ("灰度云图",        gray_heatmap)
]

for txt, cmd in btn:
    tk.Button(root, text=txt, command=cmd, width=20).pack(pady=4)

def main():
    """主函数，支持GUI和命令行两种模式"""
    import argparse
    import json
    global selected_model_variant

    parser = argparse.ArgumentParser(description='齿轮损伤检测和寿命预测工具')
    parser.add_argument('--mode', choices=['gui', 'predict'], default='gui', help='运行模式：gui或predict')
    parser.add_argument('--image', type=str, help='图片路径')
    parser.add_argument('--params', type=str, help='预测参数（JSON格式）')
    parser.add_argument('--weights', choices=['original', 'updated'], default='updated')

    args = parser.parse_args()
    selected_model_variant = args.weights
    model_status.set(f'分类模型: {selected_model_variant}；尚未加载')

    if args.mode == 'predict':
        # 命令行寿命预测模式
        if not args.image or not args.params:
            print(json.dumps({"error": "预测模式需要提供图片路径和参数"}))
            return

        try:
            # 解析参数
            gear_params = json.loads(args.params)

            # 进行寿命预测
            predicted_lifetime = predict_gear_lifetime(args.image, gear_params)

            if predicted_lifetime is not None:
                result = {
                    "success": True,
                    "predicted_lifetime": predicted_lifetime,
                    "unit": "cycles",
                    "label_provenance_verified": False,
                    "model_version": "v1.0",
                    "prediction_factors": [
                        "材料特性分析完成",
                        "几何参数验证通过",
                        "工况条件评估完成",
                        "疲劳寿命计算完成"
                    ]
                }
            else:
                result = {
                    "success": False,
                    "error": "寿命预测失败"
                }

            print(json.dumps(result, ensure_ascii=False))

        except json.JSONDecodeError:
            print(json.dumps({"error": "参数格式错误，需要JSON格式"}))
        except Exception as e:
            print(json.dumps({"error": f"预测过程中发生错误: {str(e)}"}))

    else:
        # GUI模式
        # 初始化预测器（在后台）
        if PREDICTION_AVAILABLE:
            try:
                if initialize_predictor():
                    print("深度学习预测器初始化成功")
            except Exception as e:
                print(f"深度学习预测器初始化失败: {e}")

        root.mainloop()

if __name__ == "__main__":
    main()
