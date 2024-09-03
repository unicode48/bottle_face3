from bottle import route, run, template, request
import dlib
import cv2
import numpy as np
import base64

# Dlibの顔検出器を初期化
detector = dlib.get_frontal_face_detector()

# アップロードされた画像を処理する関数
def process_image(file):
    # アップロードされたファイルを読み込み
    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 顔を検出
    faces = detector(gray)
    # 合成する画像を読み込む（アルファチャンネル付き）
    overlay_img = cv2.imread('genta.png', cv2.IMREAD_UNCHANGED)
    # 合成する画像のサイズを顔に合わせてリサイズ
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        # 元の画像の比率を保持してリサイズ
        overlay_img_resized = resize_with_aspect_ratio(overlay_img, int(w*1.3), int(h*1.3))
        # 合成画像を顔の中心に重ねる
        y_offset = int(face_center_y - overlay_img_resized.shape[0] // 2)
        x_offset = int(face_center_x - overlay_img_resized.shape[1] // 2)
        # アルファチャンネルを使用して透明な部分を正しくマスクする
        alpha_mask = overlay_img_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            image[y_offset:y_offset+overlay_img_resized.shape[0], x_offset:x_offset+overlay_img_resized.shape[1], c] = (
                alpha_mask * overlay_img_resized[:, :, c] +
                (1.0 - alpha_mask) * image[y_offset:y_offset+overlay_img_resized.shape[0], x_offset:x_offset+overlay_img_resized.shape[1], c]
            )
    # 処理した画像をbase64エンコードして返す
    _, img_encoded = cv2.imencode('.jpg', image)
    return base64.b64encode(img_encoded).decode('utf-8')

# 元の比率を保持してリサイズする関数
def resize_with_aspect_ratio(img, target_width, target_height):
    original_height, original_width = img.shape[:2]
    aspect_ratio = original_width / original_height
    if target_width / target_height > aspect_ratio:
        # 元の画像の幅を基準にリサイズ
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # 元の画像の高さを基準にリサイズ
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

# ホームページを表示するルート
@route('/')
def index():
    return '''
        <h1>元太アプリ</h1>
        <h2>みんな元太になるよ</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <text>顔が写っているファイルをえらんでね</text><br>
            <input type="file" name="file" id="imageInput" accept="image/*"><br>
            <img id="selectedImage" src="#" alt="選択された画像" style="max-width: 100%; max-height: 400px; display: none;"><br>
            <text>ボタンを押すとみんな元太になるよ</text><br>
            <input type="submit" value="実行">
        </form>
        <script>
            document.getElementById('imageInput').addEventListener('change', function(event) {
            const selectedImage = document.getElementById('selectedImage');
            const file = event.target.files[0];    
            if (file) {
                selectedImage.style.display = 'block';
                selectedImage.file = file;
                const reader = new FileReader();
                
                reader.onload = (function(aImg) { return function(e) { aImg.src = e.target.result; }; })(selectedImage);
                reader.readAsDataURL(file);
            } else {
                selectedImage.style.display = 'none';
            }
            });
        </script>
    '''

# 画像をアップロードして処理するエンドポイント
@route('/upload', method='POST')
def do_upload():
    upload = request.files.get('file')
    if upload:
        # 処理された画像を取得
        processed_image = process_image(upload)
        # 画像を表示するページにリダイレクト
        return template('<img src="data:image/jpeg;base64,{{img}}" /><br><br><button onclick="history.back()">戻る</button>', img=processed_image)
    else:
        return "No file uploaded"

if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True, reloader=True)
