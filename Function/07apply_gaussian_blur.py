import cv2
#ぼけ画像関数 by Gauussian Blur
def apply_gaussian_blur(image_path, kernel_size=5):
     """
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