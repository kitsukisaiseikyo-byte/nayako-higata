"""
干潟監視システム - GitHub Actions用 (潮位推定強化版)
- 30分ごとの自動実行
- CSV形式でデータ蓄積 (UTF-8 + Shift-JIS)
- 潮位推定機能付き (ハフ変換による直線検出採用)
- 判定精度向上
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
import math

# 日本時間のタイムゾーン
JST = timezone(timedelta(hours=9))

# --- 設定項目 ---
MAIN_CAMERA_PAGE_URL = "https://www.kitsukibousai.jp/camera.html?no=4"
BASE_IMAGE_URL = "https://www.kitsukibousai.jp"

# ROI設定 (干潟検出用 - 下部に集中)
ROI_Y_START = 270  
ROI_Y_END = 350    
ROI_X_START = 380
ROI_X_END = 630

# 潮位測定用ROI (岸壁の垂直ライン)
# ※注意: ROIの幅が狭すぎると直線検出が難しくなるため、必要に応じてXの幅を広げてください
TIDE_X_START = 500
TIDE_X_END = 550
TIDE_Y_START = 190
TIDE_Y_END = 235

# 判別パラメータ
RELATIVE_BRIGHTNESS_THRESHOLD = 0.80
SATURATION_RATIO_MAX = 0.75
BLUE_RATIO_MAX = 0.15
TEXTURE_THRESHOLD = 12
BRIGHTNESS_THRESHOLD_MIN = 70

# 出力ディレクトリ
RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CSV_FILE = os.path.join(RESULTS_DIR, "monitoring_log.csv")
CSV_FILE_SJIS = os.path.join(RESULTS_DIR, "monitoring_log_sjis.csv")
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
    岸壁のラインから潮位を推定 (ハフ変換による直線検出版)
    """
    if img is None or is_night:
        return None
    
    # ROI切り出し
    roi = img[y_start:y_end, x_start:x_end]
    
    # グレースケール変換
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. ノイズ除去 (ガウシアンブラー)
    # 波の細かい模様を消して、主要なエッジだけ残す
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # 2. エッジ検出 (Canny法)
    # 閾値は環境によって微調整が必要だが、一旦標準的な値で設定
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. 直線検出 (確率的ハフ変換)
    # minLineLength: 誤検知を減らすため、ROI幅の半分以上の長さがある線のみ採用
    roi_width = x_end - x_start
    min_line_len = roi_width * 0.4 
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=15,         # 投票数の閾値 (低いほど検出しやすいがノイズも増える)
        minLineLength=min_line_len, 
        maxLineGap=10         # 線が途切れていても許容する隙間
    )
    
    water_line_relative = None
    
    if lines is not None:
        # 検出された線の中から「水平に近い」線のみを抽出
        valid_y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 角度計算 (ラジアン -> 度)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 水平に近い線 (±15度以内) のみを採用
            if abs(angle) < 15:
                # 線のY座標の中点をリストに追加
                valid_y_coords.append((y1 + y2) / 2)
        
        if valid_y_coords:
            # 複数の線が見つかった場合、中央値を採用して外れ値を除去
            water_line_relative = np.median(valid_y_coords)
            print(f"  直線検出成功: {len(valid_y_coords)}本検出, Y={water_line_relative:.1f}")

    # 直線が見つからなかった場合のフォールバック (従来の輝度変化法)
    if water_line_relative is None:
        print("  ⚠️ 直線検出失敗 -> 輝度勾配法(バックアップ)を使用")
        vertical_profile = np.mean(gray_roi, axis=1)
        gradient = np.gradient(vertical_profile)
        water_line_relative = np.argmin(gradient)

    # 潮位計算
    water_line_absolute = y_start + water_line_relative
    tide_range = y_end - y_start
    
    # 下に行くほどY座標が大きくなるため、(1 - 比率) で水位が高いほど1.0に近づける
    tide_level_normalized = 1.0 - (water_line_relative / tide_range)
    # 範囲制限 (0.0 ~ 1.0)
    tide_level_normalized = max(0.0, min(1.0, tide_level_normalized))
    
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
        'method': 'hough_lines' if lines is not None else 'gradient_fallback'
    }

def analyze_tidal_flat(img, roi_y_start, roi_y_end, roi_x_start, roi_x_end,
                       relative_brightness_threshold, saturation_ratio_max,
                       blue_ratio_max, texture_threshold, brightness_min):
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
    blue_mask = cv2.inRange(hsv_roi, (85, 30, 30), (135, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / (roi.shape[0] * roi.shape[1])
    
    # テクスチャ分析
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_std = np.std(roi_gray)
    
    print(f"\n📊 解析結果:")
    print(f"  • ROI輝度:        {roi_brightness:.2f} / {full_brightness:.2f}")
    print(f"  • 輝度比率:       {brightness_ratio:.3f} (閾値: >{relative_brightness_threshold})")
    print(f"  • 彩度比率:       {saturation_ratio:.3f} (閾値: <{saturation_ratio_max})")
    print(f"  • 青色比率:       {blue_ratio:.3%} (閾値: <{blue_ratio_max})")
    print(f"  • テクスチャ:     {texture_std:.2f} (閾値: >{texture_threshold})")
    
    # 夜間チェック
    if roi_brightness < brightness_min:
        print(f"\n⚠️  夜間判定 (ROI輝度 {roi_brightness:.2f} < {brightness_min})")
        return {
            'is_tidal_flat': None,
            'status': "夜間(解析不可)",
            'confidence': 0,
            'brightness_ratio': brightness_ratio,
            'saturation_ratio': saturation_ratio,
            'blue_ratio': blue_ratio,
            'texture_std': texture_std,
            'roi_brightness': roi_brightness,
            'full_brightness': full_brightness,
            'is_night': True
        }
    
    if full_brightness < 60:
        print(f"\n⚠️  全体が暗すぎる (全体輝度 {full_brightness:.2f} < 60)")
        return {
            'is_tidal_flat': None,
            'status': "夜間(解析不可)",
            'confidence': 0,
            'brightness_ratio': brightness_ratio,
            'saturation_ratio': saturation_ratio,
            'blue_ratio': blue_ratio,
            'texture_std': texture_std,
            'roi_brightness': roi_brightness,
            'full_brightness': full_brightness,
            'is_night': True
        }
    
    # 判定ロジック
    conditions = []
    scores = []
    
    if brightness_ratio > relative_brightness_threshold:
        conditions.append("✓ 相対的に明るい")
        scores.append(30)
    else:
        conditions.append("✗ 明るさ不足")
        scores.append(0)
    
    if saturation_ratio < saturation_ratio_max:
        conditions.append("✓ 彩度が低い")
        scores.append(25)
    else:
        conditions.append("✗ 彩度が高い")
        scores.append(0)
    
    if blue_ratio < blue_ratio_max:
        conditions.append("✓ 青色少ない")
        scores.append(25)
    else:
        conditions.append(f"✗ 青色多い({blue_ratio:.1%})")
        scores.append(0)
    
    if texture_std > texture_threshold:
        conditions.append("✓ テクスチャ不均一")
        scores.append(20)
    else:
        conditions.append("✗ テクスチャ均一")
        scores.append(0)
    
    confidence_score = sum(scores)
    conditions_met = sum(s > 0 for s in scores)
    is_tidal_flat = conditions_met >= 3
    
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
        'full_brightness': full_brightness,
        'is_night': False
    }

def save_annotated_image(img, tidal_result, tide_result, timestamp):
    """解析結果を画像に描画して保存"""
    if img is None:
        return None
    
    img_annotated = img.copy()
    
    # 干潟ROI描画
    cv2.rectangle(img_annotated, 
                  (ROI_X_START, ROI_Y_START), 
                  (ROI_X_END, ROI_Y_END),
                  (0, 255, 0), 3)
    
    # 潮位測定ライン描画
    if tide_result:
        cv2.rectangle(img_annotated,
                      (TIDE_X_START, TIDE_Y_START),
                      (TIDE_X_END, TIDE_Y_END),
                      (255, 0, 0), 3)
        
        water_y = int(tide_result['water_line_y'])
        
        # 検出方法によって線の色を変える (ハフ変換=赤, フォールバック=オレンジ)
        line_color = (0, 0, 255) if tide_result.get('method') == 'hough_lines' else (0, 165, 255)
        
        cv2.line(img_annotated,
                 (TIDE_X_START - 30, water_y),
                 (TIDE_X_END + 30, water_y),
                 line_color, 3)
                 
        method_str = " (Auto)" if tide_result.get('method') == 'hough_lines' else " (Fallback)"
        cv2.putText(img_annotated, f"Water Surface{method_str}",
                    (TIDE_X_END + 40, water_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    
    # テキスト情報描画
    if tidal_result:
        status_map = {
            "干潟あり": "Tidal Flat: YES",
            "水面/潮位高": "Tidal Flat: NO",
            "夜間(解析不可)": "Night (No Analysis)"
        }
        status_en = status_map.get(tidal_result['status'], tidal_result['status'])
        
        text_size = cv2.getTextSize(status_en, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_annotated, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
        cv2.putText(img_annotated, status_en,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        confidence_text = f"Confidence: {tidal_result['confidence']}%"
        cv2.rectangle(img_annotated, (5, 40), (text_size[0] + 15, 65), (0, 0, 0), -1)
        cv2.putText(img_annotated, confidence_text,
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    
    if tide_result:
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
    
    # 画像保存
    filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    success = cv2.imwrite(filepath, img_annotated, encode_param)
    
    if success:
        print(f"  画像保存成功: {filepath}")
    else:
        print(f"  ⚠️ 画像保存失敗: {filepath}", file=sys.stderr)
    
    return filename

def save_to_csv(timestamp, tidal_result, tide_result, image_filename):
    """CSV形式でデータを保存"""
    headers = [
        'timestamp', 'is_tidal_flat', 'status', 'confidence',
        'brightness_ratio', 'saturation_ratio', 'blue_ratio', 'texture_std',
        'tide_level', 'tide_status', 'water_line_y', 'image_file'
    ]
    
    # 日本語→英語マッピング
    status_en_map = {
        "干潟あり": "Tidal Flat Detected",
        "水面/潮位高": "Water Surface",
        "夜間(解析不可)": "Night (No Analysis)"
    }
    tide_en_map = {
        "満潮": "High Tide", "上げ潮": "Rising Tide", "中潮": "Mid Tide",
        "下げ潮": "Falling Tide", "干潮": "Low Tide"
    }
    
    # 英語版データ (UTF-8)
    data_row_en = [
        timestamp.isoformat(),
        tidal_result['is_tidal_flat'] if tidal_result else None,
        status_en_map.get(tidal_result['status'], tidal_result['status']) if tidal_result else None,
        tidal_result['confidence'] if tidal_result else None,
        f"{tidal_result['brightness_ratio']:.3f}" if tidal_result else None,
        f"{tidal_result['saturation_ratio']:.3f}" if tidal_result else None,
        f"{tidal_result['blue_ratio']:.3f}" if tidal_result else None,
        f"{tidal_result['texture_std']:.2f}" if tidal_result else None,
        f"{tide_result['tide_level']:.3f}" if tide_result else None,
        tide_en_map.get(tide_result['tide_status'], tide_result['tide_status']) if tide_result else None,
        tide_result['water_line_y'] if tide_result else None,
        image_filename
    ]
    
    # 日本語版データ (Shift-JIS)
    data_row_ja = [
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
    ]
    
    # UTF-8保存
    csv_exists = os.path.exists(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not csv_exists: writer.writerow(headers)
            writer.writerow(data_row_en)
        print(f"  ✓ CSV(UTF-8)保存: {CSV_FILE}")
    except Exception as e:
        print(f"  ⚠️ CSV保存失敗: {e}", file=sys.stderr)
        
    # Shift-JIS保存
    csv_sjis_exists = os.path.exists(CSV_FILE_SJIS)
    try:
        with open(CSV_FILE_SJIS, 'a', newline='', encoding='shift_jis', errors='replace') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not csv_sjis_exists: writer.writerow(headers)
            writer.writerow(data_row_ja)
        print(f"  ✓ CSV(Shift-JIS)保存: {CSV_FILE_SJIS}")
    except Exception as e:
        print(f"  ⚠️ CSV(SJIS)保存失敗: {e}", file=sys.stderr)

def save_latest_json(timestamp, tidal_result, tide_result, image_filename):
    """最新結果をJSON保存"""
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
    timestamp = datetime.now(JST)
    print(f"\n{'='*70}")
    print(f"🌊 干潟監視システム実行: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
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
    
    # 干潟分析
    tidal_result = analyze_tidal_flat(
        current_image,
        ROI_Y_START, ROI_Y_END,
        ROI_X_START, ROI_X_END,
        RELATIVE_BRIGHTNESS_THRESHOLD,
        SATURATION_RATIO_MAX,
        BLUE_RATIO_MAX,
        TEXTURE_THRESHOLD,
        BRIGHTNESS_THRESHOLD_MIN
    )
    
    # 潮位推定 (ハフ変換ロジック搭載)
    is_night = tidal_result.get('is_night', False) if tidal_result else False
    tide_result = estimate_tide_level(
        current_image,
        TIDE_X_START, TIDE_X_END,
        TIDE_Y_START, TIDE_Y_END,
        is_night
    )
    
    # 結果表示
    if tidal_result:
        if tidal_result.get('is_night'):
            print(f"\n【夜間モード】解析スキップ")
        else:
            print(f"\n【干潟判定】")
            print(f"  状態: {tidal_result['status']}")
            print(f"  信頼度: {tidal_result['confidence']}/100点")
    
    if tide_result:
        print(f"\n【潮位推定】")
        print(f"  状態: {tide_result['tide_status']}")
        print(f"  潮位レベル: {tide_result['tide_level']:.1%}")
        print(f"  検出手法: {tide_result.get('method')}")
    
    # データ保存
    image_filename = save_annotated_image(current_image, tidal_result, tide_result, timestamp)
    save_to_csv(timestamp, tidal_result, tide_result, image_filename)
    save_latest_json(timestamp, tidal_result, tide_result, image_filename)
    
    print(f"\n✓ 全処理完了")
    print(f"{'='*70}\n")
