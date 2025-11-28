"""
干潟監視システム - 完全版
- 生画像 + アノテーション画像の両方を保存
- テクスチャ重視の判定ロジック
- 潮汐データ統合（海上保安庁）
- 水位測定精度改善
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

# ROI設定 (干潟検出用)
ROI_Y_START = 270  
ROI_Y_END = 350    
ROI_X_START = 380
ROI_X_END = 630

# 潮位測定用ROI (岸壁の垂直ライン)
TIDE_X_START = 500
TIDE_X_END = 550
TIDE_Y_START = 190
TIDE_Y_END = 235

# 判別パラメータ（最適化済み）
RELATIVE_BRIGHTNESS_THRESHOLD = 0.85
SATURATION_RATIO_MAX = 0.50
BLUE_RATIO_MAX = 0.05
TEXTURE_THRESHOLD = 12.00
BRIGHTNESS_THRESHOLD_MIN = 70

# 出力ディレクトリ
RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CSV_FILE = os.path.join(RESULTS_DIR, "monitoring_log.csv")
CSV_FILE_SJIS = os.path.join(RESULTS_DIR, "monitoring_log_sjis.csv")
LATEST_JSON = os.path.join(RESULTS_DIR, "latest_result.json")

# 潮汐データ設定
TIDE_DATA_FILE = "tide_prediction.json"
TIDE_DATA_CACHE_HOURS = 6

os.makedirs(IMAGES_DIR, exist_ok=True)

# ========================================
# 潮汐データ管理機能
# ========================================

def load_tide_data():
    """
    潮汐データを読み込み（キャッシュ有効期限チェック付き）
    """
    if not os.path.exists(TIDE_DATA_FILE):
        return None
    
    try:
        with open(TIDE_DATA_FILE, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        
        # 更新日時チェック
        updated_at = datetime.fromisoformat(cached['updated_at'])
        hours_ago = (datetime.now() - updated_at).total_seconds() / 3600
        
        if hours_ago > TIDE_DATA_CACHE_HOURS:
            print(f"⚠️ 潮汐データが古い（{hours_ago:.1f}時間前）")
            return None
        
        return cached['data']
    
    except Exception as e:
        print(f"⚠️ 潮汐データ読み込み失敗: {e}")
        return None


def get_current_tide_level(current_time, tide_data):
    """
    現在時刻の潮位を取得（線形補間）
    """
    if not tide_data:
        return None
    
    date_str = current_time.strftime('%Y-%m-%d')
    hour = current_time.hour
    minute = current_time.minute
    
    # 現在時刻のデータを探す
    current_hour_data = None
    next_hour_data = None
    
    for item in tide_data:
        if item['date'] == date_str:
            item_hour = int(item['time'].split(':')[0])
            
            if item_hour == hour:
                current_hour_data = item
            elif item_hour == (hour + 1) % 24:
                next_hour_data = item
    
    if not current_hour_data:
        return None
    
    # 線形補間
    if next_hour_data:
        level_current = current_hour_data['level_cm']
        level_next = next_hour_data['level_cm']
        
        interpolated = level_current + (level_next - level_current) * (minute / 60)
        return interpolated
    else:
        return current_hour_data['level_cm']


def analyze_tide_phase(current_time, tide_data):
    """
    潮汐フェーズを詳細分析
    """
    if not tide_data:
        return None
    
    date_str = current_time.strftime('%Y-%m-%d')
    
    # 当日のデータを抽出
    today_data = [item for item in tide_data if item['date'] == date_str]
    
    if not today_data:
        return None
    
    # 潮位の推移を解析
    levels = [item['level_cm'] for item in today_data]
    times = [item['time'] for item in today_data]
    
    # 満潮・干潮を検出
    high_tides = []
    low_tides = []
    
    for i in range(1, len(levels) - 1):
        # 極大値（満潮）
        if levels[i] > levels[i-1] and levels[i] > levels[i+1]:
            high_tides.append({
                'time': times[i],
                'level': levels[i],
                'hour': int(times[i].split(':')[0])
            })
        # 極小値（干潮）
        elif levels[i] < levels[i-1] and levels[i] < levels[i+1]:
            low_tides.append({
                'time': times[i],
                'level': levels[i],
                'hour': int(times[i].split(':')[0])
            })
    
    # 現在の潮位
    current_level = get_current_tide_level(current_time, tide_data)
    
    # 現在の状態を判定
    current_hour = current_time.hour
    phase = 'unknown'
    nearest_low = None
    time_from_low = None
    
    # 最も近い干潮を探す
    for low in low_tides:
        hour_diff = abs(current_hour - low['hour'])
        if hour_diff <= 2:  # 前後2時間以内
            phase = 'low'
            nearest_low = low
            time_from_low = current_hour - low['hour']
            break
    
    # 満潮判定
    if phase == 'unknown':
        for high in high_tides:
            hour_diff = abs(current_hour - high['hour'])
            if hour_diff <= 2:
                phase = 'high'
                break
    
    # 上げ潮・下げ潮判定
    if phase == 'unknown' and len(today_data) >= 3:
        current_idx = current_hour
        if current_idx > 0 and current_idx < len(levels) - 1:
            if levels[current_idx] > levels[current_idx - 1]:
                phase = 'rising'
            else:
                phase = 'falling'
    
    return {
        'phase': phase,
        'current_level': current_level,
        'high_tides': high_tides,
        'low_tides': low_tides,
        'nearest_low': nearest_low,
        'time_from_low': time_from_low
    }


# ========================================
# 基本機能
# ========================================

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

# ========================================
# 干潟判定（テクスチャ重視＋潮汐統合）
# ========================================

def analyze_tidal_flat_with_tide(img, roi_y_start, roi_y_end, roi_x_start, roi_x_end,
                                  current_time, brightness_min=70):
    """
    テクスチャ重視＋潮汐データ併用の干潟判定
    """
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
    
    # 輝度チェック
    roi_brightness = np.mean(hsv_roi[:,:,2])
    
    if roi_brightness < brightness_min:
        return {
            'is_tidal_flat': None,
            'status': "夜間(解析不可)",
            'confidence': 0,
            'brightness_ratio': 0,
            'saturation_ratio': 0,
            'blue_ratio': 0,
            'texture_std': 0,
            'roi_brightness': roi_brightness,
            'full_brightness': 0,
            'tide_phase': 'night',
            'tide_level': None,
            'is_night': True
        }
    
    # テクスチャ計算（最重要指標）
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_std = np.std(roi_gray)
    
    # 参考指標
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    full_brightness = np.mean(hsv_full[:,:,2])
    brightness_ratio = roi_brightness / (full_brightness + 0.001)
    
    roi_saturation = np.mean(hsv_roi[:,:,1])
    full_saturation = np.mean(hsv_full[:,:,1])
    saturation_ratio = roi_saturation / (full_saturation + 0.001)
    
    blue_mask = cv2.inRange(hsv_roi, (85, 30, 30), (135, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / (roi.shape[0] * roi.shape[1])
    
    # 潮汐データ取得・解析
    tide_data = load_tide_data()
    tide_info = analyze_tide_phase(current_time, tide_data)
    
    print(f"\n📊 統合判定:")
    print(f"  • テクスチャ: {texture_std:.2f} (主指標)")
    print(f"  • 輝度比率: {brightness_ratio:.3f}")
    print(f"  • 青色比率: {blue_ratio:.3f}")
    
    if tide_info:
        print(f"  • 潮汐フェーズ: {tide_info['phase']}")
        if tide_info['current_level']:
            print(f"  • 現在潮位: {tide_info['current_level']:.0f}cm")
    
    # ========================================
    # 判定ロジック（テクスチャ＋潮汐）
    # ========================================
    
    TEXTURE_HIGH = 15.0
    TEXTURE_VERY_HIGH = 20.0
    
    is_tidal_flat = False
    confidence = 0
    status = "水面/潮位高"
    
    # ケース1: 満潮時の異常値除外
    if tide_info and tide_info['phase'] == 'high':
        if texture_std > TEXTURE_HIGH:
            is_tidal_flat = False
            confidence = 85
            status = "水面/潮位高"
            print(f"  判定: 満潮時の波・反射（texture={texture_std:.1f}）")
        else:
            is_tidal_flat = False
            confidence = 95
            status = "水面/潮位高"
            print(f"  判定: 満潮時＆低テクスチャ → 水面確定")
    
    # ケース2: 干潮時＋高テクスチャ → 干潟確定
    elif tide_info and tide_info['phase'] == 'low':
        if texture_std > TEXTURE_HIGH:
            is_tidal_flat = True
            confidence = 95
            status = "干潟あり"
            print(f"  判定: 干潟あり（干潮時＋高テクスチャ）")
        else:
            is_tidal_flat = False
            confidence = 70
            status = "水面/潮位高"
            print(f"  判定: 干潮時だが低テクスチャ")
    
    # ケース3: 干潮後1-3時間
    elif tide_info and tide_info['time_from_low'] and 0 < tide_info['time_from_low'] <= 3:
        if texture_std > TEXTURE_HIGH:
            is_tidal_flat = True
            confidence = 90
            status = "干潟あり"
            print(f"  判定: 干潟あり（干潮{tide_info['time_from_low']}時間後）")
        else:
            is_tidal_flat = False
            confidence = 60
            status = "水面/潮位高"
            print(f"  判定: 干潮後だが低テクスチャ")
    
    # ケース4: テクスチャのみで判定
    elif texture_std > TEXTURE_VERY_HIGH:
        is_tidal_flat = True
        confidence = 80
        status = "干潟あり"
        if blue_ratio < 0.05:
            confidence = 85
        print(f"  判定: 干潟あり（極めて高いテクスチャ）")
    
    elif texture_std > TEXTURE_HIGH:
        if blue_ratio < 0.05 and brightness_ratio > 0.80:
            is_tidal_flat = True
            confidence = 70
            status = "干潟あり"
            print(f"  判定: 干潟あり（テクスチャ＋補助指標）")
        else:
            is_tidal_flat = False
            confidence = 65
            status = "水面/潮位高"
            print(f"  判定: テクスチャ中程度")
    
    else:
        is_tidal_flat = False
        confidence = 90
        status = "水面/潮位高"
        print(f"  判定: 水面確定（低テクスチャ）")
    
    return {
        'is_tidal_flat': is_tidal_flat,
        'status': status,
        'confidence': confidence,
        'brightness_ratio': brightness_ratio,
        'saturation_ratio': saturation_ratio,
        'blue_ratio': blue_ratio,
        'texture_std': texture_std,
        'roi_brightness': roi_brightness,
        'full_brightness': full_brightness,
        'tide_phase': tide_info['phase'] if tide_info else 'unknown',
        'tide_level': tide_info['current_level'] if tide_info else None,
        'is_night': False
    }


# ========================================
# 水位測定（改善版）
# ========================================

def estimate_tide_level_improved(img, x_start, x_end, y_start, y_end, is_night=False):
    """
    改善版潮位推定（複数手法統合）
    """
    if img is None or is_night:
        return None
    
    roi = img[y_start:y_end, x_start:x_end]
    roi_height, roi_width = roi.shape[:2]
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 前処理強化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # エッジ検出
    edges = cv2.Canny(denoised, 80, 200, apertureSize=3)
    
    # ハフ変換
    min_line_len = roi_width * 0.5
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=min_line_len, maxLineGap=15)
    
    water_line_hough = None
    hough_confidence = 0
    
    if lines is not None and len(lines) > 0:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 10 and length > roi_width * 0.3:
                y_mid = (y1 + y2) / 2
                horizontal_lines.append({
                    'y': y_mid,
                    'length': length,
                    'angle': abs(angle)
                })
        
        if horizontal_lines:
            for line_info in horizontal_lines:
                length_score = line_info['length'] / roi_width
                angle_score = 1 - (line_info['angle'] / 10)
                line_info['score'] = length_score * angle_score
            
            best_line = max(horizontal_lines, key=lambda x: x['score'])
            water_line_hough = best_line['y']
            hough_confidence = min(100, int(best_line['score'] * 100))
    
    # 輝度勾配法（バックアップ）
    vertical_profile = np.mean(enhanced, axis=1)
    gradient = np.gradient(vertical_profile)
    gradient_abs = np.abs(gradient)
    
    water_line_gradient = None
    if np.max(gradient_abs) > 0:
        candidates = []
        top_n = min(5, len(gradient_abs))
        top_indices = np.argsort(gradient_abs)[-top_n:]
        
        for idx in top_indices:
            if roi_height * 0.2 < idx < roi_height * 0.8:
                candidates.append(idx)
        
        if candidates:
            water_line_gradient = candidates[0]
    
    # 統合判定
    if water_line_hough is not None:
        water_line_relative = water_line_hough
        final_confidence = hough_confidence
        detection_method = 'hough'
    elif water_line_gradient is not None:
        water_line_relative = water_line_gradient
        final_confidence = 50
        detection_method = 'gradient'
    else:
        water_line_relative = roi_height / 2
        final_confidence = 10
        detection_method = 'fallback'
    
    # 潮位計算
    water_line_absolute = y_start + water_line_relative
    tide_range = y_end - y_start
    tide_level_normalized = 1.0 - (water_line_relative / tide_range)
    tide_level_normalized = max(0.0, min(1.0, tide_level_normalized))
    
    # 状態判定
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
        'method': detection_method,
        'confidence': final_confidence
    }

# ========================================
# 画像保存（生画像＋アノテーション）
# ========================================

def save_images(img, tidal_result, tide_result, timestamp):
    """
    生画像とアノテーション画像の両方を保存
    """
    if img is None:
        return None, None
    
    # 生画像保存
    raw_filename = f"raw_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    raw_filepath = os.path.join(IMAGES_DIR, raw_filename)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    success_raw = cv2.imwrite(raw_filepath, img, encode_param)
    
    if success_raw:
        print(f"  ✓ 生画像保存: {raw_filepath}")
    
    # アノテーション画像作成
    img_annotated = img.copy()
    
    # 干潟ROI描画
    cv2.rectangle(img_annotated, (ROI_X_START, ROI_Y_START), 
                  (ROI_X_END, ROI_Y_END), (0, 255, 0), 3)
    
    # 潮位測定ライン描画
    if tide_result:
        cv2.rectangle(img_annotated, (TIDE_X_START, TIDE_Y_START),
                      (TIDE_X_END, TIDE_Y_END), (255, 0, 0), 3)
        
        water_y = int(tide_result['water_line_y'])
        confidence = tide_result.get('confidence', 50)
        
        if confidence >= 70:
            line_color = (0, 255, 0)
        elif confidence >= 40:
            line_color = (0, 165, 255)
        else:
            line_color = (0, 0, 255)
        
        cv2.line(img_annotated, (TIDE_X_START - 30, water_y),
                 (TIDE_X_END + 30, water_y), line_color, 3)
    
    # テキスト描画
    if tidal_result:
        status_map = {
            "干潟あり": "Tidal Flat: YES",
            "水面/潮位高": "Tidal Flat: NO",
            "夜間(解析不可)": "Night (No Analysis)"
        }
        status_en = status_map.get(tidal_result['status'], tidal_result['status'])
        
        cv2.rectangle(img_annotated, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(img_annotated, status_en, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        confidence_text = f"Confidence: {tidal_result['confidence']}%"
        cv2.rectangle(img_annotated, (5, 40), (200, 65), (0, 0, 0), -1)
        cv2.putText(img_annotated, confidence_text, (10, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if tide_result:
        tide_map = {
            "満潮": "High Tide", "上げ潮": "Rising", "中潮": "Mid Tide",
            "下げ潮": "Falling", "干潮": "Low Tide"
        }
        tide_en = tide_map.get(tide_result['tide_status'], tide_result['tide_status'])
        tide_conf = tide_result.get('confidence', 0)
        tide_text = f"Tide: {tide_en} ({tide_result['tide_level']:.0%}) [{tide_conf}%]"
        
        cv2.rectangle(img_annotated, (5, 70), (400, 95), (0, 0, 0), -1)
        
        if tide_conf >= 70:
            tide_color = (0, 255, 0)
        elif tide_conf >= 40:
            tide_color = (255, 200, 0)
        else:
            tide_color = (0, 0, 255)
        
        cv2.putText(img_annotated, tide_text, (10, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tide_color, 2)
    
    # タイムスタンプ
    time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S JST")
    cv2.rectangle(img_annotated, (5, img_annotated.shape[0] - 30),
                  (350, img_annotated.shape[0] - 5), (0, 0, 0), -1)
    cv2.putText(img_annotated, time_text, (10, img_annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # アノテーション画像保存
    annotated_filename = f"annotated_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated_filepath = os.path.join(IMAGES_DIR, annotated_filename)
    success_annotated = cv2.imwrite(annotated_filepath, img_annotated, encode_param)
    
    if success_annotated:
        print(f"  ✓ アノテーション画像保存: {annotated_filepath}")
    
    return raw_filename, annotated_filename


# ========================================
# CSV保存
# ========================================

def save_to_csv(timestamp, tidal_result, tide_result, image_filename):
    """CSV形式でデータを保存"""
    headers = [
        'timestamp', 'is_tidal_flat', 'status', 'confidence',
        'brightness_ratio', 'saturation_ratio', 'blue_ratio', 'texture_std',
        'tide_level', 'tide_status', 'tide_confidence', 'water_line_y',
        'tide_method', 'tide_phase', 'image_file'
    ]
    
    status_en_map = {
        "干潟あり": "Tidal Flat Detected",
        "水面/潮位高": "Water Surface",
        "夜間(解析不可)": "Night (No Analysis)"
    }
    tide_en_map = {
        "満潮": "High Tide", "上げ潮": "Rising Tide", "中潮": "Mid Tide",
        "下げ潮": "Falling Tide", "干潮": "Low Tide"
    }
    
    data_row = [
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
        tide_result.get('confidence', 0) if tide_result else None,
        tide_result['water_line_y'] if tide_result else None,
        tide_result.get('method', '') if tide_result else None,
        tidal_result.get('tide_phase', 'unknown') if tidal_result else 'unknown',
        image_filename
    ]
    
    csv_exists = os.path.exists(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not csv_exists:
                writer.writerow(headers)
            writer.writerow(data_row)
        print(f"  ✓ CSV保存成功")
    except Exception as e:
        print(f"  ⚠️ CSV保存失敗: {e}", file=sys.stderr)


def save_latest_json(timestamp, tidal_result, tide_result, image_filename):
    """最新結果をJSON保存"""
    latest_data = {
        'timestamp': timestamp.isoformat(),
        'tidal_flat': {
            'detected': bool(tidal_result['is_tidal_flat']) if tidal_result and tidal_result['is_tidal_flat'] is not None else None,
            'status': tidal_result['status'] if tidal_result else None,
            'confidence': int(tidal_result['confidence']) if tidal_result else None,
            'tide_phase': tidal_result.get('tide_phase', 'unknown') if tidal_result else 'unknown'
        },
        'tide': {
            'level': float(tide_result['tide_level']) if tide_result else None,
            'status': tide_result['tide_status'] if tide_result else None,
            'water_line_y': int(tide_result['water_line_y']) if tide_result else None,
            'confidence': tide_result.get('confidence', 0) if tide_result else 0,
            'method': tide_result.get('method', '') if tide_result else ''
        },
        'image_file': image_filename
    }
    
    with open(LATEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(latest_data, f, ensure_ascii=False, indent=2)


# ========================================
# メイン処理
# ========================================

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
    
    # 干潟分析（テクスチャ重視＋潮汐統合）
    tidal_result = analyze_tidal_flat_with_tide(
        current_image,
        ROI_Y_START, ROI_Y_END,
        ROI_X_START, ROI_X_END,
        timestamp,
        BRIGHTNESS_THRESHOLD_MIN
    )
    
    # 潮位推定
    is_night = tidal_result.get('is_night', False) if tidal_result else False
    tide_result = estimate_tide_level_improved(
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
            print(f"  潮汐フェーズ: {tidal_result.get('tide_phase', 'unknown')}")
    
    if tide_result:
        print(f"\n【潮位推定】")
        print(f"  状態: {tide_result['tide_status']}")
        print(f"  潮位レベル: {tide_result['tide_level']:.1%}")
        print(f"  検出手法: {tide_result.get('method')}")
        print(f"  信頼度: {tide_result.get('confidence', 0)}%")
    
    # データ保存
    raw_filename, annotated_filename = save_images(current_image, tidal_result, tide_result, timestamp)
    save_to_csv(timestamp, tidal_result, tide_result, raw_filename)
    save_latest_json(timestamp, tidal_result, tide_result, raw_filename)
    
    print(f"\n✓ 全処理完了")
    print(f"{'='*70}\n")
