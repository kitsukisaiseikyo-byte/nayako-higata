"""
干潟監視システム - GitHub Actions用
- 30分ごとの自動実行
- CSV形式でデータ蓄積
- 潮位推定機能付き
"""

import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin
import os
import sys
import csv
import json

# 日本時間のタイムゾーン
JST = timezone(timedelta(hours=9))

# --- 設定項目 ---
MAIN_CAMERA_PAGE_URL = "https://www.kitsukibousai.jp/camera.html?no=4"
BASE_IMAGE_URL = "https://www.kitsukibousai.jp"

# ROI設定 (干潟検出用)
ROI_Y_START = 200
ROI_Y_END = 350
ROI_X_START = 380
ROI_X_END = 630

# 潮位測定用ROI (岸壁の垂直ライン)
# 画像の上から下まで走査し、水面との境界を検出
TIDE_X_START = 500  # 岸壁の左端
TIDE_X_END = 550    # 岸壁の右端
TIDE_Y_START = 190  # 走査開始位置(上)
TIDE_Y_END = 235    # 走査終了位置(下)

# 判別パラメータ
RELATIVE_BRIGHTNESS_THRESHOLD = 0.85
SATURATION_RATIO_MAX = 0.85
BLUE_RATIO_MAX = 0.30
BRIGHTNESS_THRESHOLD_MIN = 70
SATURATION_MAX = 50

# 出力ディレクトリ
RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CSV_FILE = os.path.join(RESULTS_DIR, "monitoring_log.csv")
CSV_FILE_SJIS = os.path.join(RESULTS_DIR, "monitoring_log_sjis.csv")  # Shift-JIS版
LATEST_JSON = os.path.join(RESULTS_DIR, "latest_result.json")

os.makedirs(IMAGES_DIR, exist_ok=True)

# --- 関数定義 ---

def get_latest_image_url(main_page_url, base_image_url):
    """ライブカメラのメインページから最新の画像URLを取得"""
    try:
        response = requests.get(main_page_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error accessing main camera page: {e}", file=sys.stderr)
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    img_tag = soup.find('img', src=lambda s: s and 'cam_' in s)

    if img_tag:
        relative_image_url = img_tag.get('src')
        if relative_image_url:
            return urljoin(base_image_url, relative_image_url)
    return None

def download_image(url):
    """画像をダウンロード"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        np_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}", file=sys.stderr)
        return None

def estimate_tide_level(img, x_start, x_end, y_start, y_end, is_night=False):
    """
    岸壁の垂直ラインから潮位を推定
    
    原理:
    1. 岸壁の指定領域を上から下にスキャン
    2. 明度の急激な変化点を水面として検出
    3. 水面の高さ(Y座標)を潮位の指標とする
    
    夜間は解析をスキップ
    """
    if img is None or is_night:
        return None
    
    img_height = img.shape[0]
    
    # ROI領域を切り出し
    tide_roi = img[y_start:y_end, x_start:x_end]
    
    # グレースケール化
    gray_roi = cv2.cvtColor(tide_roi, cv2.COLOR_BGR2GRAY)
    
    # 垂直方向の平均輝度プロファイルを計算
    vertical_profile = np.mean(gray_roi, axis=1)
    
    # 勾配を計算(明度の変化率)
    gradient = np.gradient(vertical_profile)
    
    # 最大の負の勾配(明→暗への急変)を水面として検出
    # 水面より上は明るい(岸壁)、下は暗い(水面)
    water_line_relative = np.argmin(gradient)
    water_line_absolute = y_start + water_line_relative
    
    # 潮位レベルを正規化 (0.0=最低水位, 1.0=最高水位)
    tide_range = y_end - y_start
    tide_level_normalized = 1.0 - (water_line_relative / tide_range)
    
    # 潮位を5段階で分類
    if tide_level_normalized > 0.8:
        tide_status = "満潮"
    elif tide_level_normalized > 0.6:
        tide_status = "上げ潮"
    elif tide_level_normalized > 0.4:
        tide_status = "中潮"
    elif tide_level_normalized > 0.2:
        tide_status = "下げ潮"
    else:
        tide_status = "干潮"
    
    return {
        'water_line_y': water_line_absolute,
        'tide_level': tide_level_normalized,
        'tide_status': tide_status,
        'vertical_profile': vertical_profile.tolist()
    }

def analyze_tidal_flat(img, roi_y_start, roi_y_end, roi_x_start, roi_x_end,
                      relative_brightness_threshold, saturation_ratio_max,
                      blue_ratio_max, brightness_min, saturation_max):
    """干潟判別分析"""
    if img is None:
        return None
    
    img_height, img_width = img.shape[:2]
    
    y_start = min(max(0, roi_y_start), img_height)
    y_end = min(max(0, roi_y_end), img_height)
    x_start = min(max(0, roi_x_start), img_width)
    x_end = min(max(0, roi_x_end), img_width)
    
    if y_start >= y_end or x_start >= x_end:
        return None
    
    roi = img[y_start:y_end, x_start:x_end]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 輝度・彩度分析
    roi_brightness = np.mean(hsv_roi[:,:,2])
    full_brightness = np.mean(hsv_full[:,:,2])
    brightness_ratio = roi_brightness / (full_brightness + 0.001)
    
    roi_saturation = np.mean(hsv_roi[:,:,1])
    full_saturation = np.mean(hsv_full[:,:,1])
    saturation_ratio = roi_saturation / (full_saturation + 0.001)
    
    # 青色比率
    blue_mask = cv2.inRange(hsv_roi, (100, 50, 50), (130, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / (roi.shape[0] * roi.shape[1])
    
    # テクスチャ分析
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_std = np.std(roi_gray)
    
    # 判定ロジック
    scores = []
    scores.append(30 if brightness_ratio > relative_brightness_threshold else 0)
    scores.append(25 if saturation_ratio < saturation_ratio_max else 0)
    scores.append(25 if blue_ratio < blue_ratio_max else 0)
    scores.append(20 if texture_std > 15 else 0)
    
    if roi_brightness < brightness_min:
        is_tidal_flat = False
        confidence_score = 0
    else:
        confidence_score = sum(scores)
        is_tidal_flat = sum(s > 0 for s in scores) >= 3
    
    status = "干潟あり" if is_tidal_flat else "水面/潮位高"
    
    return {
        'is_tidal_flat': is_tidal_flat,
        'status': status,
        'confidence': confidence_score,
        'brightness_ratio': brightness_ratio,
        'saturation_ratio': saturation_ratio,
        'blue_ratio': blue_ratio,
        'texture_std': texture_std,
        'roi_brightness': roi_brightness,
        'full_brightness': full_brightness
    }

def save_annotated_image(img, tidal_result, tide_result, timestamp):
    """解析結果を画像に描画して保存"""
    if img is None:
        return None
    
    img_annotated = img.copy()
    
    # 干潟ROIを描画
    cv2.rectangle(img_annotated, 
                  (ROI_X_START, ROI_Y_START), 
                  (ROI_X_END, ROI_Y_END),
                  (0, 255, 0), 3)
    
    # ROIラベル
    cv2.putText(img_annotated, "Tidal Flat ROI",
                (ROI_X_START, ROI_Y_START - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 潮位測定ラインを描画
    if tide_result:
        cv2.rectangle(img_annotated,
                      (TIDE_X_START, TIDE_Y_START),
                      (TIDE_X_END, TIDE_Y_END),
                      (255, 0, 0), 3)
        
        # 潮位測定ラベル
        cv2.putText(img_annotated, "Tide Level",
                    (TIDE_X_START, TIDE_Y_START - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 水面ラインを描画
        water_y = int(tide_result['water_line_y'])
        cv2.line(img_annotated,
                 (TIDE_X_START - 30, water_y),
                 (TIDE_X_END + 30, water_y),
                 (0, 0, 255), 3)
        
        # 水面ラインのラベル
        cv2.putText(img_annotated, "Water Surface",
                    (TIDE_X_END + 40, water_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 判定結果を英語で表示
    if tidal_result:
        # 日本語→英語変換
        status_map = {
            "干潟あり": "Tidal Flat: YES",
            "水面/潮位高": "Tidal Flat: NO",
            "夜間(解析不可)": "Night (No Analysis)"
        }
        status_en = status_map.get(tidal_result['status'], tidal_result['status'])
        
        # 背景付きテキスト
        text_size = cv2.getTextSize(status_en, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_annotated, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
        cv2.putText(img_annotated, status_en,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        # 信頼度
        confidence_text = f"Confidence: {tidal_result['confidence']}%"
        cv2.rectangle(img_annotated, (5, 40), (text_size[0] + 15, 65), (0, 0, 0), -1)
        cv2.putText(img_annotated, confidence_text,
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    
    if tide_result:
        # 潮位状態を英語で表示
        tide_map = {
            "満潮": "High Tide",
            "上げ潮": "Rising",
            "中潮": "Mid Tide",
            "下げ潮": "Falling",
            "干潮": "Low Tide"
        }
        tide_en = tide_map.get(tide_result['tide_status'], tide_result['tide_status'])
        tide_text = f"Tide: {tide_en} ({tide_result['tide_level']:.0%})"
        
        text_size2 = cv2.getTextSize(tide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_annotated, (5, 70), (text_size2[0] + 15, 95), (0, 0, 0), -1)
        cv2.putText(img_annotated, tide_text,
                    (10, 87), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2)
    
    # タイムスタンプ
    time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S JST")
    cv2.rectangle(img_annotated, (5, img_annotated.shape[0] - 30),
                  (350, img_annotated.shape[0] - 5), (0, 0, 0), -1)
    cv2.putText(img_annotated, time_text,
                (10, img_annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 画像保存 (JPEG品質を明示的に指定)
    filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    
    # JPEG保存パラメータを指定
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    success = cv2.imwrite(filepath, img_annotated, encode_param)
    
    if success:
        print(f"  画像保存成功: {filepath}")
    else:
        print(f"  ⚠️ 画像保存失敗: {filepath}", file=sys.stderr)
    
    return filename

def save_to_csv(timestamp, tidal_result, tide_result, image_filename):
    """CSV形式でデータを保存"""
    csv_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)  # すべてクォートで囲む
        
        if not csv_exists:
            # ヘッダー行
            writer.writerow([
                'timestamp',
                'is_tidal_flat',
                'status',
                'confidence',
                'brightness_ratio',
                'saturation_ratio',
                'blue_ratio',
                'texture_std',
                'tide_level',
                'tide_status',
                'water_line_y',
                'image_file'
            ])
        
        # データ行
        writer.writerow([
            timestamp.isoformat(),
            tidal_result['is_tidal_flat'] if tidal_result else None,
            tidal_result['status'] if tidal_result else None,
            tidal_result['confidence'] if tidal_result else None,
            f"{tidal_result['brightness_ratio']:.3f}" if tidal_result else None,
            f"{tidal_result['saturation_ratio']:.3f}" if tidal_result else None,
            f"{tidal_result['blue_ratio']:.3f}" if tidal_result else None,
            f"{tidal_result['texture_std']:.2f}" if tidal_result else None,
            f"{tide_result['tide_level']:.3f}" if tide_result else None,
            tide_result['tide_status'] if tide_result else None,
            tide_result['water_line_y'] if tide_result else None,
            image_filename
        ])

def save_latest_json(timestamp, tidal_result, tide_result, image_filename):
    """最新の結果をJSON形式で保存"""
    latest_data = {
        'timestamp': timestamp.isoformat(),
        'tidal_flat': {
            'detected': bool(tidal_result['is_tidal_flat']) if tidal_result and tidal_result['is_tidal_flat'] is not None else None,
            'status': tidal_result['status'] if tidal_result else None,
            'confidence': int(tidal_result['confidence']) if tidal_result else None
        },
        'tide': {
            'level': float(tide_result['tide_level']) if tide_result else None,
            'status': tide_result['tide_status'] if tide_result else None,
            'water_line_y': int(tide_result['water_line_y']) if tide_result else None
        },
        'image_file': image_filename
    }
    
    with open(LATEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(latest_data, f, ensure_ascii=False, indent=2)

# --- メイン処理 ---
if __name__ == "__main__":
    # 日本時間を取得
    timestamp = datetime.now(JST)
    print(f"\n{'='*70}")
    print(f"🌊 干潟監視システム実行: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 1. 画像取得
    latest_url = get_latest_image_url(MAIN_CAMERA_PAGE_URL, BASE_IMAGE_URL)
    if not latest_url:
        print("✗ 画像URL取得失敗", file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ 画像URL: {latest_url}")
    
    current_image = download_image(latest_url)
    if current_image is None:
        print("✗ 画像ダウンロード失敗", file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ 画像ダウンロード成功")
    
    # 2. 干潟分析
    tidal_result = analyze_tidal_flat(
        current_image,
        ROI_Y_START, ROI_Y_END,
        ROI_X_START, ROI_X_END,
        RELATIVE_BRIGHTNESS_THRESHOLD,
        SATURATION_RATIO_MAX,
        BLUE_RATIO_MAX,
        BRIGHTNESS_THRESHOLD_MIN,
        SATURATION_MAX
    )
    
    # 3. 潮位推定
    is_night = tidal_result.get('is_night', False) if tidal_result else False
    tide_result = estimate_tide_level(
        current_image,
        TIDE_X_START, TIDE_X_END,
        TIDE_Y_START, TIDE_Y_END,
        is_night
    )
    
    # 4. 結果表示
    if tidal_result:
        if tidal_result.get('is_night'):
            print(f"\n【夜間モード】")
            print(f"  解析をスキップしました")
            print(f"  ROI輝度: {tidal_result['roi_brightness']:.2f} (閾値: {BRIGHTNESS_THRESHOLD_MIN})")
        else:
            print(f"\n【干潟判定】")
            print(f"  状態: {tidal_result['status']}")
            print(f"  信頼度: {tidal_result['confidence']}/100点")
            print(f"  輝度比率: {tidal_result['brightness_ratio']:.3f}")
            print(f"  テクスチャ: {tidal_result['texture_std']:.2f}")
    
    if tide_result:
        print(f"\n【潮位推定】")
        print(f"  状態: {tide_result['tide_status']}")
        print(f"  潮位レベル: {tide_result['tide_level']:.1%}")
        print(f"  水面位置(Y座標): {tide_result['water_line_y']}")
    elif not is_night:
        print(f"\n【潮位推定】")
        print(f"  潮位推定に失敗しました")
    
    # 5. データ保存
    image_filename = save_annotated_image(current_image, tidal_result, tide_result, timestamp)
    save_to_csv(timestamp, tidal_result, tide_result, image_filename)
    save_latest_json(timestamp, tidal_result, tide_result, image_filename)
    
    print(f"\n✓ データ保存完了")
    print(f"  - CSV: {CSV_FILE}")
    print(f"  - 画像: {os.path.join(IMAGES_DIR, image_filename)}")
    print(f"  - JSON: {LATEST_JSON}")
    
    print(f"\n{'='*70}\n")
