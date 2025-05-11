import cv2
#減色関数(量子化)by k-means
def quantize_colors(image_path, num_colors=16):
    """
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