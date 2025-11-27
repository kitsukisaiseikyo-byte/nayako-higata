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
RELATIVE_BRIGHTNESS_THRESHOLD = 0.85
SATURATION_RATIO_MAX = 0.50
BLUE_RATIO_MAX = 0.05
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

def estimate_tide_level_improved(img, x_start, x_end, y_start, y_end, is_night=False):
    """
    改善版潮位推定
    - より堅牢なエッジ検出
    - 複数手法の組み合わせ
    - 信頼度スコア付き
    """
    if img is None or is_night:
        return None
    
    # ROI切り出し
    roi = img[y_start:y_end, x_start:x_end]
    roi_height, roi_width = roi.shape[:2]
    
    # グレースケール変換
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # =========================================
    # 手法1: 改善版ハフ変換
    # =========================================
    
    # 前処理の強化
    # 1. CLAHE（コントラスト適応型ヒストグラム均等化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    
    # 2. バイラテラルフィルタ（エッジを保持しながらノイズ除去）
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. エッジ検出（Cannyのパラメータを調整）
    # 水面のエッジは比較的明瞭なので閾値を高めに設定
    edges = cv2.Canny(denoised, 80, 200, apertureSize=3)
    
    # 4. モルフォロジー処理（エッジの連続性を強化）
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # 5. ハフ変換（パラメータを調整）
    min_line_len = roi_width * 0.5  # 横幅の半分以上の線のみ
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20,              # 投票数閾値を上げて誤検知削減
        minLineLength=min_line_len, 
        maxLineGap=15              # 隙間許容を広げる
    )
    
    water_line_hough = None
    hough_confidence = 0
    
    if lines is not None and len(lines) > 0:
        # 水平線の候補を抽出（±10度以内）
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 線の長さ
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 角度計算
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 水平に近い（±10度）かつ十分長い線のみ
            if abs(angle) < 10 and length > roi_width * 0.3:
                y_mid = (y1 + y2) / 2
                horizontal_lines.append({
                    'y': y_mid,
                    'length': length,
                    'angle': abs(angle)
                })
        
        if horizontal_lines:
            # 長さと角度でスコアリング
            for line_info in horizontal_lines:
                # スコア = 長さの比率 × (1 - 角度の比率)
                length_score = line_info['length'] / roi_width
                angle_score = 1 - (line_info['angle'] / 10)
                line_info['score'] = length_score * angle_score
            
            # 最高スコアの線を採用
            best_line = max(horizontal_lines, key=lambda x: x['score'])
            water_line_hough = best_line['y']
            hough_confidence = min(100, int(best_line['score'] * 100))
            
            print(f"  [ハフ変換] {len(horizontal_lines)}本の水平線検出")
            print(f"    最良線: Y={water_line_hough:.1f}, 信頼度={hough_confidence}%")
    
    # =========================================
    # 手法2: 輝度勾配法（改善版）
    # =========================================
    
    # 横方向の平均輝度プロファイル
    vertical_profile = np.mean(enhanced, axis=1)
    
    # 勾配計算（Sobelでより堅牢に）
    gradient = np.gradient(vertical_profile)
    
    # 勾配の絶対値が大きい場所（境界候補）
    gradient_abs = np.abs(gradient)
    
    # 上位N個の候補を取得
    top_n = min(5, len(gradient_abs))
    top_indices = np.argsort(gradient_abs)[-top_n:]
    
    # 中央付近のものを優先（上端・下端は除外）
    valid_candidates = []
    for idx in top_indices:
        # ROIの上下20%は除外（ノイズが多いため）
        if roi_height * 0.2 < idx < roi_height * 0.8:
            valid_candidates.append({
                'y': idx,
                'gradient': gradient_abs[idx],
                # 中央に近いほど高スコア
                'center_score': 1 - abs(idx - roi_height/2) / (roi_height/2)
            })
    
    water_line_gradient = None
    gradient_confidence = 0
    
    if valid_candidates:
        # 勾配の強さと中央寄りのバランスでスコアリング
        for cand in valid_candidates:
            gradient_norm = cand['gradient'] / np.max(gradient_abs)
            cand['score'] = gradient_norm * 0.7 + cand['center_score'] * 0.3
        
        best_candidate = max(valid_candidates, key=lambda x: x['score'])
        water_line_gradient = best_candidate['y']
        gradient_confidence = min(100, int(best_candidate['score'] * 100))
        
        print(f"  [輝度勾配] 候補{len(valid_candidates)}箇所")
        print(f"    最良点: Y={water_line_gradient:.1f}, 信頼度={gradient_confidence}%")
    
    # =========================================
    # 手法3: 色相変化検出（新規追加）
    # =========================================
    
    # HSV変換
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 色相（H）の縦方向変化を見る
    hue_profile = np.mean(hsv_roi[:, :, 0], axis=1)
    hue_gradient = np.abs(np.gradient(hue_profile))
    
    # 彩度（S）の変化も見る
    sat_profile = np.mean(hsv_roi[:, :, 1], axis=1)
    sat_gradient = np.abs(np.gradient(sat_profile))
    
    # 色相と彩度の変化を統合
    color_gradient = (hue_gradient + sat_gradient) / 2
    
    # 最大変化点を水面境界とする
    if np.max(color_gradient) > 5:  # ノイズレベル以上
        water_line_color = np.argmax(color_gradient)
        # ROIの上下20%は除外
        if roi_height * 0.2 < water_line_color < roi_height * 0.8:
            color_confidence = min(100, int(np.max(color_gradient) / 50 * 100))
            print(f"  [色相変化] Y={water_line_color:.1f}, 信頼度={color_confidence}%")
        else:
            water_line_color = None
            color_confidence = 0
    else:
        water_line_color = None
        color_confidence = 0
    
    # =========================================
    # 統合判定（複数手法の加重平均）
    # =========================================
    
    candidates = []
    
    if water_line_hough is not None:
        candidates.append({
            'y': water_line_hough,
            'confidence': hough_confidence,
            'method': 'hough'
        })
    
    if water_line_gradient is not None:
        candidates.append({
            'y': water_line_gradient,
            'confidence': gradient_confidence,
            'method': 'gradient'
        })
    
    if water_line_color is not None:
        candidates.append({
            'y': water_line_color,
            'confidence': color_confidence,
            'method': 'color'
        })
    
    if not candidates:
        print("  ⚠️ 全手法で検出失敗 → フォールバック")
        water_line_relative = roi_height / 2  # 中央をデフォルト
        final_confidence = 10
        detection_method = 'fallback'
    else:
        # 信頼度による加重平均
        total_weight = sum(c['confidence'] for c in candidates)
        
        if total_weight > 0:
            weighted_y = sum(c['y'] * c['confidence'] for c in candidates) / total_weight
            water_line_relative = weighted_y
            final_confidence = int(total_weight / len(candidates))
            
            # 使用した手法を記録
            methods_used = [c['method'] for c in candidates]
            detection_method = '+'.join(methods_used)
            
            print(f"  [統合判定] Y={water_line_relative:.1f}")
            print(f"    使用手法: {detection_method}")
            print(f"    最終信頼度: {final_confidence}%")
            
            # 外れ値チェック（候補間の差が大きすぎる場合は警告）
            if len(candidates) >= 2:
                y_values = [c['y'] for c in candidates]
                y_std = np.std(y_values)
                if y_std > roi_height * 0.2:
                    print(f"    ⚠️ 手法間のバラツキ大（標準偏差={y_std:.1f}）")
                    final_confidence = max(10, final_confidence - 30)
        else:
            water_line_relative = candidates[0]['y']
            final_confidence = candidates[0]['confidence']
            detection_method = candidates[0]['method']
    
    # 潮位計算
    water_line_absolute = y_start + water_line_relative
    tide_range = y_end - y_start
    
    # 正規化（0.0～1.0）
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
        'confidence': final_confidence,  # 新規追加
        'candidates': len(candidates)     # 新規追加
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

def save_images(img, tidal_result, tide_result, timestamp):
    """
    生画像とアノテーション画像の両方を保存
    """
    if img is None:
        return None, None
    
    # ----------------------------------------
    # 1. 生画像保存（CNN訓練用）
    # ----------------------------------------
    raw_filename = f"raw_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    raw_filepath = os.path.join(IMAGES_DIR, raw_filename)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    success_raw = cv2.imwrite(raw_filepath, img, encode_param)
    
    if success_raw:
        print(f"  ✓ 生画像保存: {raw_filepath}")
    else:
        print(f"  ⚠️ 生画像保存失敗", file=sys.stderr)
    
    # ----------------------------------------
    # 2. アノテーション画像保存（確認用）
    # ----------------------------------------
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
        
        # 信頼度に応じて色を変更
        confidence = tide_result.get('confidence', 50)
        if confidence >= 70:
            line_color = (0, 255, 0)  # 緑（高信頼度）
        elif confidence >= 40:
            line_color = (0, 165, 255)  # オレンジ（中信頼度）
        else:
            line_color = (0, 0, 255)  # 赤（低信頼度）
        
        cv2.line(img_annotated,
                 (TIDE_X_START - 30, water_y),
                 (TIDE_X_END + 30, water_y),
                 line_color, 3)
        
        # 信頼度表示
        method_str = f" ({tide_result.get('method', 'unknown')})"
        cv2.putText(img_annotated, f"Water Surface{method_str}",
                    (TIDE_X_END + 40, water_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
    
    # 干潟判定テキスト
    if tidal_result:
        status_map = {
            "干潟あり": "Tidal Flat: YES",
            "水面/潮位高": "Tidal Flat: NO",
            "夜間(解析不可)": "Night (No Analysis)"
        }
        status_en = status_map.get(tidal_result['status'], tidal_result['status'])
        
        # 背景
        text_size = cv2.getTextSize(status_en, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_annotated, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
        
        # テキスト
        cv2.putText(img_annotated, status_en,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        # 干潟判定の信頼度
        confidence_text = f"Confidence: {tidal_result['confidence']}%"
        cv2.rectangle(img_annotated, (5, 40), (text_size[0] + 15, 65), (0, 0, 0), -1)
        cv2.putText(img_annotated, confidence_text,
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    
    # 潮位情報
    if tide_result:
        tide_map = {
            "満潮": "High Tide", "上げ潮": "Rising", "中潮": "Mid Tide",
            "下げ潮": "Falling", "干潟": "Low Tide", "干潮": "Low Tide"
        }
        tide_en = tide_map.get(tide_result['tide_status'], tide_result['tide_status'])
        
        # 信頼度も表示
        tide_conf = tide_result.get('confidence', 0)
        tide_text = f"Tide: {tide_en} ({tide_result['tide_level']:.0%}) [{tide_conf}%]"
        
        text_size2 = cv2.getTextSize(tide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_annotated, (5, 70), (text_size2[0] + 15, 95), (0, 0, 0), -1)
        
        # 信頼度で色分け
        if tide_conf >= 70:
            tide_color = (0, 255, 0)  # 緑
        elif tide_conf >= 40:
            tide_color = (255, 200, 0)  # 黄色
        else:
            tide_color = (0, 0, 255)  # 赤
        
        cv2.putText(img_annotated, tide_text,
                    (10, 87), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, tide_color, 2)
    
    # タイムスタンプ
    time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S JST")
    cv2.rectangle(img_annotated, (5, img_annotated.shape[0] - 30),
                  (350, img_annotated.shape[0] - 5), (0, 0, 0), -1)
    cv2.putText(img_annotated, time_text,
                (10, img_annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # アノテーション画像保存
    annotated_filename = f"annotated_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated_filepath = os.path.join(IMAGES_DIR, annotated_filename)
    success_annotated = cv2.imwrite(annotated_filepath, img_annotated, encode_param)
    
    if success_annotated:
        print(f"  ✓ アノテーション画像保存: {annotated_filepath}")
    
    return raw_filename, annotated_filename


# ========================================
# CSV保存関数に信頼度カラム追加
# ========================================

def save_to_csv(timestamp, tidal_result, tide_result, image_filename):
    """CSV保存（信頼度情報追加版）"""
    headers = [
        'timestamp', 'is_tidal_flat', 'status', 'confidence',
        'brightness_ratio', 'saturation_ratio', 'blue_ratio', 'texture_std',
        'tide_level', 'tide_status', 'tide_confidence', 'water_line_y',  # tide_confidence追加
        'tide_method', 'image_file'  # tide_method追加
    ]
    
    # 英語版データ
    status_en_map = {
        "干潟あり": "Tidal Flat Detected",
        "水面/潮位高": "Water Surface",
        "夜間(解析不可)": "Night (No Analysis)"
    }
    tide_en_map = {
        "満潮": "High Tide", "上げ潮": "Rising Tide", "中潮": "Mid Tide",
        "下げ潮": "Falling Tide", "干潟": "Low Tide", "干潮": "Low Tide"
    }
    
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
        tide_result.get('confidence', 0) if tide_result else None,  # 追加
        tide_result['water_line_y'] if tide_result else None,
        tide_result.get('method', '') if tide_result else None,  # 追加
        image_filename
    ]
    
    # UTF-8保存
    csv_exists = os.path.exists(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not csv_exists: 
                writer.writerow(headers)
            writer.writerow(data_row_en)
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
    
    # 潮位推定（改善版に変更）
    is_night = tidal_result.get('is_night', False) if tidal_result else False
    tide_result = estimate_tide_level_improved(  # ← 関数名変更
        current_image,
        TIDE_X_START, TIDE_X_END,
        TIDE_Y_START, TIDE_Y_END,
        is_night
    )
    
    # 結果表示（信頼度追加）
    if tide_result:
        print(f"\n【潮位推定】")
        print(f"  状態: {tide_result['tide_status']}")
        print(f"  潮位レベル: {tide_result['tide_level']:.1%}")
        print(f"  検出手法: {tide_result.get('method')}")
        print(f"  信頼度: {tide_result.get('confidence', 0)}%")  # 追加
    
    # 画像保存（2ファイル版に変更）
    raw_filename, annotated_filename = save_images(  # ← 関数名変更
        current_image, tidal_result, tide_result, timestamp
    )
    
    # CSV保存（生画像のファイル名を使用）
    save_to_csv(timestamp, tidal_result, tide_result, raw_filename)
    
    # JSON保存も更新
    save_latest_json(timestamp, tidal_result, tide_result, raw_filename)
    
    print(f"\n✓ 全処理完了")
    print(f"{'='*70}\n")
