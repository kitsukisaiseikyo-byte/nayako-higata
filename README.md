# 🌊 干潟監視システム

杵築市納屋港付近のライブカメラ画像を15分ごとに自動解析し、干潟の出現状況と潮位を記録するシステムです。

## 📊 機能

- **干潟検出**: 画像解析により干潟の出現を自動判定
- **潮位推定**: 岸壁の水面高さから潮位レベルを推定
- **データ蓄積**: CSV形式で解析結果を時系列に保存
- **画像保存**: 解析結果を注釈付きで画像として保存
- **自動実行**: GitHub Actionsで15分ごとに自動実行

## 🚀 セットアップ手順

### 1. リポジトリの作成

```bash
# 新規リポジトリを作成
mkdir tidal-flat-monitor
cd tidal-flat-monitor
git init
```

### 2. ファイルの配置

以下のファイルをリポジトリに配置してください:

```
tidal-flat-monitor/
├── .github/
│   └── workflows/
│       └── monitor.yml          # GitHub Actions設定
├── monitor_tidal_flat.py        # メインスクリプト
├── requirements.txt             # Python依存関係
├── README.md                    # このファイル
└── results/                     # 自動生成されるディレクトリ
    ├── images/                  # 画像保存先
    ├── monitoring_log.csv       # データログ
    └── latest_result.json       # 最新結果
```

### 3. GitHubにプッシュ

```bash
git add .
git commit -m "Initial commit: 干潟監視システム"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tidal-flat-monitor.git
git push -u origin main
```

### 4. GitHub Actions設定

1. リポジトリの **Settings** → **Actions** → **General** を開く
2. "Workflow permissions"で **Read and write permissions** を選択
3. "Allow GitHub Actions to create and approve pull requests" にチェック
4. **Save** をクリック

### 5. 動作確認

1. リポジトリの **Actions** タブを開く
2. "干潟監視システム" ワークフローを選択
3. **Run workflow** ボタンで手動実行してテスト
4. 成功すると`results/`ディレクトリにデータが保存されます

## 📈 データ形式

### CSV出力 (monitoring_log.csv)

| 列名 | 説明 | 例 |
|------|------|-----|
| timestamp | 実行日時 | 2025-11-18T12:52:00 |
| is_tidal_flat | 干潟検出 | True/False |
| status | 判定結果 | 干潟あり/水面/潮位高 |
| confidence | 信頼度 | 70 (0-100点) |
| brightness_ratio | 輝度比率 | 0.85 |
| saturation_ratio | 彩度比率 | 0.80 |
| blue_ratio | 青色比率 | 0.00 |
| texture_std | テクスチャ標準偏差 | 17.60 |
| tide_level | 潮位レベル | 0.45 (0.0-1.0) |
| tide_status | 潮位状態 | 干潮/下げ潮/中潮/上げ潮/満潮 |
| water_line_y | 水面Y座標 | 280 |
| image_file | 画像ファイル名 | capture_20251118_125200.jpg |

### JSON出力 (latest_result.json)

```json
{
  "timestamp": "2025-11-18T12:52:00",
  "tidal_flat": {
    "detected": true,
    "status": "干潟あり",
    "confidence": 70
  },
  "tide": {
    "level": 0.45,
    "status": "干潟",
    "water_line_y": 280
  },
  "image_file": "capture_20251118_125200.jpg"
}
```

## 🔧 パラメータ調整

`monitor_tidal_flat.py`の以下のパラメータで精度を調整できます:

```python
# 干潟検出ROI
ROI_Y_START = 200
ROI_Y_END = 350
ROI_X_START = 380
ROI_X_END = 630

# 潮位測定ROI (岸壁部分)
TIDE_X_START = 440
TIDE_X_END = 480
TIDE_Y_START = 150
TIDE_Y_END = 350

# 判別閾値
RELATIVE_BRIGHTNESS_THRESHOLD = 0.85  # 輝度比率
SATURATION_RATIO_MAX = 0.85           # 彩度比率
BLUE_RATIO_MAX = 0.30                 # 青色比率
```

## 📊 データ分析例

### Pythonでの分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み
df = pd.read_csv('results/monitoring_log.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 潮位の時系列グラフ
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['tide_level'])
plt.title('潮位変化')
plt.xlabel('時刻')
plt.ylabel('潮位レベル')
plt.grid(True)
plt.show()

# 干潟出現率の集計
print(df['is_tidal_flat'].value_counts())
```

## 🛠️ トラブルシューティング

### ワークフローが実行されない
- GitHub Actionsの権限設定を確認
- cronの時刻がUTCで指定されていることを確認

### 画像がコミットされない
- `.gitignore`に`results/`が含まれていないか確認
- リポジトリのストレージ容量を確認

### 判定精度が低い
- ROIパラメータを実際の画像に合わせて調整
- 複数の時間帯でテストして閾値を微調整

## 📝 ライセンス

MIT License

## 🤝 貢献

Issue・Pull Requestを歓迎します!

---

**ライブカメラ提供**: 杵築市防災情報
