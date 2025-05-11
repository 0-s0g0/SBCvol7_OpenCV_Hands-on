import cv2
import numpy as np
import matplotlib.pyplot as plt

# 提供されたドット絵化関数 (そのまま使用)
def pixelate_image(image_path, dot_size=10):
    """
    画像をドット絵風に変換する関数
    入力:
        image_path (str): 入力画像ファイルのパス
        dot_size (int): 元の画像の何ピクセルを1ドットとするか
    出力:
        numpy.ndarray or None: ドット絵化された画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    height, width = img.shape[:2]
    new_width = width // dot_size
    new_height = height // dot_size

    if new_width <= 0 or new_height <= 0:
         print(f"エラー: dot_size が大きすぎます。画像サイズ ({width}x{height}) よりも小さい値を指定してください。")
         return None

    img_small = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    img_dot = cv2.resize(img_small, (width, height), interpolation=cv2.INTER_NEAREST)

    return img_dot

# --- 新しい画像処理関数 ---

def convert_to_grayscale(image_path):
    """
    画像をグレースケールに変換する関数
    入力:
        image_path (str): 入力画像ファイルのパス
    出力:
        numpy.ndarray or None: グレースケール画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # BGR画像をグレースケールに変換
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img

def binarize_image(image_path, threshold=128):
    """
    画像を二値化する関数 (指定した閾値で)
    入力:
        image_path (str): 入力画像ファイルのパス
        threshold (int): 閾値 (0-255)。この値より大きいピクセルは白、小さいピクセルは黒になる
    出力:
        numpy.ndarray or None: 二値化された画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # グレースケールで読み込む
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # cv2.threshold を使って二値化
    # ret は計算された閾値 (cv2.THRESH_OTSU などを使った場合)、binary_img が結果の画像
    ret, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return binary_img

def binarize_image_otsu(image_path):
    """
    画像を大津の方法で二値化する関数 (自動的に閾値を決定)
    入力:
        image_path (str): 入力画像ファイルのパス
    出力:
        numpy.ndarray or None: 二値化された画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # グレースケールで読み込む
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # 大津の方法で閾値を自動計算し、二値化
    ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f"大津の方法で計算された閾値: {ret}")

    return binary_img


def quantize_colors(image_path, num_colors=16):
    """
    画像の色の数を減らす (量子化) 関数 (K-Meansを使用)
    入力:
        image_path (str): 入力画像ファイルのパス
        num_colors (int): 量子化後の色の数
    出力:
        numpy.ndarray or None: 色が量子化された画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # 画像をピクセルのリストに変換 (高さ * 幅, チャンネル数)
    pixels = np.float32(img.reshape(-1, 3))

    # K-Meansの停止条件を定義 (EPS: 精度, MAX_ITER: 最大繰り返し回数)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-Meansを実行
    # compactness: 各点から重心までの距離の二乗の合計
    # labels: 各ピクセルがどのクラスターに属するか
    # centers: 各クラスターの中心 (代表色)
    compactness, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 中心 (代表色) をuint8に変換
    centers = np.uint8(centers)

    # 各ピクセルを対応する代表色に置き換える
    quantized_pixels = centers[labels.flatten()]

    # 元の画像サイズに戻す
    quantized_img = quantized_pixels.reshape(img.shape)

    return quantized_img

# --- 面白い関数例 ---

def apply_canny_edge_detection(image_path, lower_threshold=100, upper_threshold=200):
    """
    Cannyエッジ検出を適用する関数
    入力:
        image_path (str): 入力画像ファイルのパス
        lower_threshold (int): 閾値1
        upper_threshold (int): 閾値2
    出力:
        numpy.ndarray or None: エッジ画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # グレースケールで読み込む
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # ノイズを減らすためにぼかしを適用 (オプションだが推奨)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0) # カーネルサイズは奇数

    # Cannyエッジ検出を適用
    edges = cv2.Canny(img_blur, lower_threshold, upper_threshold)

    return edges

def apply_gaussian_blur(image_path, kernel_size=5):
     """
     ガウシアンブラーを適用する関数
     入力:
         image_path (str): 入力画像ファイルのパス
         kernel_size (int): ぼかしのカーネルサイズ (奇数である必要がある)
     出力:
         numpy.ndarray or None: ぼかしが適用された画像 (NumPy配列)、失敗した場合はNone
     """
     img = cv2.imread(image_path)
     if img is None:
         print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
         return None

     if kernel_size % 2 == 0:
         print(f"エラー: kernel_size は奇数である必要があります。現在の値: {kernel_size}")
         return None

     # ガウシアンブラーを適用
     blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

     return blurred_img

def apply_sepia_filter(image_path):
    """
    画像をセピア調に変換する関数
    入力:
        image_path (str): 入力画像ファイルのパス
    出力:
        numpy.ndarray or None: セピア調の画像 (NumPy配列)、失敗した場合はNone
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル '{image_path}' が見つからないか、読み込めません。パスを確認してください。")
        return None

    # セピア変換行列 (BGR順)
    # R = (R * 0.393) + (G * 0.769) + (B * 0.189)
    # G = (R * 0.349) + (G * 0.686) + (B * 0.168)
    # B = (R * 0.272) + (G * 0.534) + (B * 0.131)
    sepia_matrix = np.array([[0.131, 0.534, 0.272],
                             [0.168, 0.686, 0.349],
                             [0.189, 0.769, 0.393]]).T # OpenCVはBGRなので行列を転置

    # 画像の各ピクセルに変換行列を適用
    # マトリクス積を計算するためにピクセルを浮動小数点に変換
    img_float = np.float32(img)
    sepia_img = cv2.transform(img_float, sepia_matrix)

    # 値を0-255の範囲にクリップし、uint8に戻す
    sepia_img = np.clip(sepia_img, 0, 255)
    sepia_img = np.uint8(sepia_img)

    return sepia_img


# --- 関数の使い方例 ---

# ダミーの画像ファイルを準備 (実際には適切なパスに置き換えてください)
# 例: 'test_image.jpg' というファイル名で、このスクリプトと同じディレクトリにあると仮定
# もし画像がない場合は、適当な画像を準備するか、以下のパスを変更してください。
input_image_path = '/lena_std.bmp' # <-- ここをあなたの画像ファイルパスに書き換えてください

# Matplotlibで画像を表示するヘルパー関数
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

print(f"処理対象の画像ファイル: {input_image_path}")
print("-" * 20)

# 1. ドット絵化の例 (提供コードの再掲)
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

# 6. ガウシアンブラーの例
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