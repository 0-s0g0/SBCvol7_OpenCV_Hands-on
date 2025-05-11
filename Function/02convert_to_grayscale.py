import cv2
#グレースケール変換
def convert_to_grayscale(image_path):
    """
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