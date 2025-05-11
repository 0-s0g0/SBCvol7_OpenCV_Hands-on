import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像処理関数のインポート
import pixelate_image
import convert_to_grayscale
import binarize_image 
import binarize_image_otsu 
import quantize_colors
import apply_canny_edge_detection
import apply_gaussian_blur
import apply_sepia_filter


# 画像を表示する関数 
def show_image(img, title="Image"):
    """画像をMatplotlibで表示するヘルパー関数"""
    if img is None:
        print(f"表示する画像がありません: {title}")
        return

    # 画像のチャンネル数を確認
    if len(img.shape) == 3: # カラー画像 (BGRまたはRGB)
        # OpenCVはBGRで読み込むため、Matplotlibで表示するためにRGBに変換
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_display)
    elif len(img.shape) == 2: # グレースケールまたは二値画像
        # グレースケールとして表示
        plt.imshow(img, cmap='gray')
    else:
        print(f"不明な画像形式です。表示できません: {title}")
        return

    plt.title(title)
    plt.axis('off') # 軸非表示
    plt.show()


# 関数呼び出し

# 画像ファイル指定
input_image_path = './lena_std.bmp' # <-- ここをあなたの画像ファイルパスに書き換えてください

print(f"処理対象の画像ファイル: {input_image_path}")
print("-" * 20)

# 1. ドット絵化の例
dot_pixel_size = 15
dot_image = pixelate_image(input_image_path, dot_pixel_size)
show_image(dot_image, f'Pixelated Image (Dot Size: {dot_pixel_size})')
if dot_image is not None:
    cv2.imwrite('output_dot.png', dot_image)
    print("ドット絵画像を 'output_dot.png' に保存しました。")
print("-" * 20)

# 2. グレースケール変換の例
gray_image = convert_to_grayscale(input_image_path)
show_image(gray_image, 'Grayscale Image')
if gray_image is not None:
    cv2.imwrite('output_grayscale.png', gray_image)
    print("グレースケール画像を 'output_grayscale.png' に保存しました。")
print("-" * 20)

# 3. 二値化の例 (固定閾値)
threshold_value = 100 # 閾値を調整してみてください
binary_image_fixed = binarize_image(input_image_path, threshold_value)
show_image(binary_image_fixed, f'Binarized Image (Threshold: {threshold_value})')
if binary_image_fixed is not None:
    cv2.imwrite('output_binary_fixed.png', binary_image_fixed)
    print("固定閾値で二値化画像を 'output_binary_fixed.png' に保存しました。")
print("-" * 20)

# 3. 二値化の例 (大津の方法)
binary_image_otsu = binarize_image_otsu(input_image_path)
show_image(binary_image_otsu, 'Binarized Image (Otsu\'s Method)')
if binary_image_otsu is not None:
    cv2.imwrite('output_binary_otsu.png', binary_image_otsu)
    print("大津の方法で二値化画像を 'output_binary_otsu.png' に保存しました。")
print("-" * 20)

# 4. 色の量子化の例
num_colors_to_keep = 8 # 色数を調整してみてください (例: 4, 8, 16, 32...)
quantized_image = quantize_colors(input_image_path, num_colors_to_keep)
show_image(quantized_image, f'Color Quantized Image ({num_colors_to_keep} colors)')
if quantized_image is not None:
    cv2.imwrite(f'output_quantized_{num_colors_to_keep}.png', quantized_image)
    print(f"色が量子化された画像を 'output_quantized_{num_colors_to_keep}.png' に保存しました。")
print("-" * 20)

# 5. エッジ検出の例 (Canny)
edges_image = apply_canny_edge_detection(input_image_path, 50, 150) # 閾値を調整してみてください
show_image(edges_image, 'Canny Edge Detection')
if edges_image is not None:
    cv2.imwrite('output_edges.png', edges_image)
    print("エッジ検出画像を 'output_edges.png' に保存しました。")
print("-" * 20)

# 6. ガウシアン平滑化の例
blur_kernel = 9 # ぼかしの強さを調整 (3, 5, 7, 9... 奇数)
blurred_image = apply_gaussian_blur(input_image_path, blur_kernel)
show_image(blurred_image, f'Gaussian Blur (Kernel: {blur_kernel})')
if blurred_image is not None:
    cv2.imwrite(f'output_blurred_{blur_kernel}.png', blurred_image)
    print(f"ぼかし画像を 'output_blurred_{blur_kernel}.png' に保存しました。")
print("-" * 20)

# 7. セピアフィルターの例
sepia_image = apply_sepia_filter(input_image_path)
show_image(sepia_image, 'Sepia Filter')
if sepia_image is not None:
    cv2.imwrite('output_sepia.png', sepia_image)
    print("セピア画像を 'output_sepia.png' に保存しました。")
print("-" * 20)

print("すべての処理が完了しました。出力ファイルを確認してください。")