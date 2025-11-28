name: 干潟監視システム（潮汐データ統合版）

on:
  schedule:
    # 15分ごとに実行（JST 6:00-18:45）
    # 00, 15, 30, 45分に実行
    - cron: '0,15,30,45 21-23 * * *'    # JST 6:00-7:45
    - cron: '0,15,30,45 0-9 * * *'      # JST 9:00-18:45
    
    # 毎朝3時に潮汐データ更新（JST 12:00 = UTC 3:00）
    - cron: '0 3 * * *'
    
  workflow_dispatch: # 手動実行も可能

jobs:
  # ジョブ1: 潮汐データ更新（1日1回）
  update-tide-data:
    runs-on: ubuntu-latest
    # 毎朝3時のみ実行
    if: github.event.schedule == '0 3 * * *' || github.event_name == 'workflow_dispatch'
    
    steps:
    - name: リポジトリをチェックアウト
      uses: actions/checkout@v3
      
    - name: Pythonセットアップ
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: 依存関係をインストール
      run: |
        pip install requests beautifulsoup4
        
    - name: 潮汐データを取得
      run: |
        python fetch_tide_data.py
        
    - name: 潮汐データをコミット
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add tide_prediction.json
        git diff --quiet && git diff --staged --quiet || git commit -m "🌊 潮汐データ更新: $(TZ=Asia/Tokyo date +'%Y-%m-%d')"
        
    - name: 変更をプッシュ
      uses: ad-m/github-push-action@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        branch: ${{ github.ref }}

  # ジョブ2: 干潟監視（15分ごと）
  monitor-tidal-flat:
    runs-on: ubuntu-latest
    # 潮汐データ更新以外の時間帯に実行
    if: github.event.schedule != '0 3 * * *' || github.event_name == 'workflow_dispatch'
    
    steps:
    - name: リポジトリをチェックアウト
      uses: actions/checkout@v3
      
    - name: Pythonセットアップ
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: 依存関係をインストール
      run: |
        pip install requests beautifulsoup4 opencv-python-headless numpy
        
    - name: 干潟解析を実行
      run: |
        python monitor_tidal_flat.py
        
    - name: 解析結果をコミット
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add results/
        git diff --quiet && git diff --staged --quiet || git commit -m "🌊 解析結果更新: $(TZ=Asia/Tokyo date +'%Y-%m-%d %H:%M:%S JST')"
        
    - name: 変更をプッシュ
      uses: ad-m/github-push-action@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        branch: ${{ github.ref }}
