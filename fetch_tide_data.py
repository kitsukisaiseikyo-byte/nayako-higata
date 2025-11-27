"""
潮汐データ取得専用スクリプト
GitHub Actionsで毎朝実行して最新の潮汐予測を取得
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta

# --- 設定 ---
AREA_CODE = "4419"      # 国東港の地域コード
BACK_PARAM = "3"
DAYS_TO_FETCH = 7       # 7日分取得
FILE_NAME = "tide_prediction.json"

def fetch_tide_data():
    """
    海上保安庁から潮汐データを取得
    """
    all_tide_data = []
    start_date = datetime.now().date()
    BASE_URL = "https://www1.kaiho.mlit.go.jp/TIDE/pred2/cgi-bin/TidePredCgi.cgi"
    
    print(f"🌊 潮汐予測データ取得開始（国東港, {DAYS_TO_FETCH}日分）")
    print(f"取得日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    for i in range(DAYS_TO_FETCH):
        target_date = start_date + timedelta(days=i)
        
        params = {
            'area': AREA_CODE,
            'back': BACK_PARAM,
            'year': target_date.strftime('%Y'),
            'month': target_date.strftime('%m'),
            'day': target_date.strftime('%d')
        }
        
        current_date_fmt = target_date.strftime('%Y-%m-%d')
        print(f"  取得中: {current_date_fmt}", end="")
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            
            target_table = soup.find('table', bgcolor="#e3ffe3")
            
            if target_table:
                rows = target_table.find_all('tr')
                
                # 0-11時のデータ
                hours_0_11 = [td.text.strip() for td in rows[0].find_all('td')[1:]]
                levels_0_11 = [td.text.strip() for td in rows[1].find_all('td')[1:]]
                
                # 12-23時のデータ
                hours_12_23 = [td.text.strip() for td in rows[2].find_all('td')[1:]]
                levels_12_23 = [td.text.strip() for td in rows[3].find_all('td')[1:]]
                
                hours = hours_0_11 + hours_12_23
                levels = levels_0_11 + levels_12_23
                
                day_count = 0
                for j in range(24):
                    time_str = f"{hours[j].zfill(2)}:00"
                    level_cm = levels[j].replace(' ', '')
                    
                    all_tide_data.append({
                        "date": current_date_fmt,
                        "time": time_str,
                        "level_cm": int(level_cm)
                    })
                    day_count += 1
                
                print(f" ✓ ({day_count}件)")
            else:
                print(f" ✗ テーブル特定失敗")
                
        except Exception as e:
            print(f" ✗ エラー: {e}")
    
    print("-" * 70)
    print(f"合計 {len(all_tide_data)}件のデータを取得")
    
    return all_tide_data


def save_tide_data(data):
    """
    潮汐データをJSONファイルに保存
    """
    output = {
        'updated_at': datetime.now().isoformat(),
        'source': '海上保安庁 国東港',
        'area_code': AREA_CODE,
        'data': data
    }
    
    with open(FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ ファイルに保存: {FILE_NAME}")


def analyze_tide_summary(data):
    """
    取得したデータの簡易サマリーを表示
    """
    if not data:
        return
    
    print("\n📊 潮汐データサマリー:")
    print("-" * 70)
    
    # 日別の統計
    dates = sorted(set(item['date'] for item in data))
    
    for date in dates[:3]:  # 最初の3日分を表示
        day_data = [item for item in data if item['date'] == date]
        levels = [item['level_cm'] for item in day_data]
        
        max_level = max(levels)
        min_level = min(levels)
        max_time = day_data[levels.index(max_level)]['time']
        min_time = day_data[levels.index(min_level)]['time']
        
        print(f"\n  📅 {date}")
        print(f"    満潮: {max_time} ({max_level}cm)")
        print(f"    干潮: {min_time} ({min_level}cm)")
        print(f"    潮位差: {max_level - min_level}cm")
    
    if len(dates) > 3:
        print(f"\n  ... 他 {len(dates) - 3}日分のデータ")


def main():
    """
    メイン処理
    """
    print("=" * 70)
    print("🌊 潮汐データ自動取得システム")
    print("=" * 70)
    print()
    
    # データ取得
    tide_data = fetch_tide_data()
    
    if tide_data:
        # ファイル保存
        save_tide_data(tide_data)
        
        # サマリー表示
        analyze_tide_summary(tide_data)
        
        print("\n" + "=" * 70)
        print("🎉 潮汐データの取得・保存が完了しました！")
        print("=" * 70)
    else:
        print("\n⚠️ データを取得できませんでした")
        print("ネットワーク接続とパラメータを確認してください")
        exit(1)


if __name__ == "__main__":
    main()
