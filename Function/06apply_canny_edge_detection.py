import cv2

#エッジを検出する関数 bycanny
def apply_canny_edge_detection(image_path, lower_threshold=100, upper_threshold=200):
    """
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