# jointCameraTimeAnalysis

input 配下の時系列CSVから、矩形ROIごとの平均値・標準偏差・分散を計算し、ROI位置やヒートマップの時間変化も可視化するための実装です。

## 前提

- 各CSVは先頭7行がメタ情報
- 8行目が x 座標ヘッダ
- 9行目以降が y ごとの温度ヒートマップ
- ファイル名の末尾インデックスを時系列順として扱う

## Python 環境構築

### 1. Python のインストール

Python 3.12 以上が必要です。  
[https://www.python.org/downloads/](https://www.python.org/downloads/) から最新版をダウンロードし、インストールしてください。

インストール時に **「Add Python to PATH」にチェック** を入れてください。

インストール後、バージョンを確認します。

```powershell
python --version
# Python 3.12.x と表示されればOK
```

### 2. リポジトリのクローン

```powershell
git clone https://github.com/yok1012/jointCameraTimeAnalysis.git
cd jointCameraTimeAnalysis
```

### 3. 仮想環境の作成と有効化

```powershell
# 仮想環境を作成
python -m venv .venv

# 仮想環境を有効化（PowerShell の場合）
.venv\Scripts\Activate.ps1

# コマンドプロンプトの場合
.venv\Scripts\activate.bat
```

> **注**: PowerShell で実行ポリシーエラーが出る場合は、先に以下を実行してください。
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. 依存パッケージのインストール

```powershell
pip install -r requirements.txt
```

### 5. 動作確認

```powershell
# Streamlit UI を起動
python -m streamlit run app.py
```

ブラウザで `http://localhost:8501` が開けば環境構築は完了です。

## ROIをJSONで指定して分析

```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py analyze --input-dir input --roi-file rois.example.json --output-dir output --time-axis index --invert-y
```

出力:

- output/roi_timeseries.csv
- output/roi_timeseries.png
- output/roi_overlay_frame_001.png
- output/roi_crops_frame_001.png

## ROIをヒートマップ上で選択

```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py select-rois --input-dir input --frame-index 1 --output rois.json --invert-y
```

既存ROIを重ねて再編集したい場合:

```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py select-rois --input-dir input --frame-index 1 --output rois.json --existing-roi-file rois.json --preview-output roi_preview.png --invert-y
```

操作:

- マウスドラッグで矩形を選択
- a キーで現在の矩形をROIとして追加
- u キーで最後のROIを取り消し
- s キーで保存して終了
- q キーで保存せず終了

出力:

- rois.json
- roi_preview.png
- roi_preview_crops.png

## ヒートマップの時間変化を出力

```powershell
.venv\Scripts\python.exe analyze_heatmap_rois.py render-heatmaps --input-dir input --output-dir heatmap_output --roi-file rois.json --invert-y
```

出力:

- heatmap_output/frames/heatmap_001.png などの連番画像
- heatmap_output/heatmap_summary.png
- heatmap_output/heatmap_animation.gif

補足:

- ROIファイルを指定すると、各ヒートマップに領域枠と名前を重ねて出力
- 色スケールは全フレーム共通なので、時間変化を見比べやすい
- --frame-step 2 のようにすると間引いて出力可能
- --invert-y を付けるとヒートマップの y 軸を反転可能

## ROI JSON形式

```json
[
  {
    "name": "roi_1",
    "x_min": 464,
    "x_max": 470,
    "y_min": 154,
    "y_max": 160
  }
]
```

ROIは最大30件までです。

## Streamlit ブラウザ UI

コマンドラインの代わりに、ブラウザで操作できる UI も用意しています。

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

起動後、ブラウザで `http://localhost:8501` を開いてください。

### 5 タブ構成

| タブ | 機能 |
|------|------|
| ① データセット設定 | 入力ディレクトリ・出力先・データセット名・カラーマップ・y軸反転などを設定し CSV を読み込む |
| ② ROI設定 | ヒートマップ上でドラッグして矩形ROIを追加・編集・削除。グリッド分割一括追加・切り出しオフセット・有効/無効切替・X/Y方向反転にも対応 |
| ③ 分析実行 | ROI統計を計算し、時系列グラフ・基準ROIとの差分グラフ・オーバーレイ画像・クロップ画像を表示。分析結果の保存も可能 |
| ④ ヒートマップ可視化 | 全フレームのヒートマップ PNG と GIF アニメーションを生成。フレームブラウザで個別閲覧・ダウンロード可能 |
| ⑤ 保存済み分析 | 過去に保存した分析結果の一覧表示・読み込み・グラフ再描画・ROI定義の復元・削除 |

### 操作手順（全体の流れ）

#### タブ① データセット設定

1. **入力ディレクトリを指定** — CSVファイルが格納されているフォルダパスを入力する
   - 過去に読み込んだディレクトリはプルダウンから選択可能（セッション中自動保存、最大20件）
2. **「CSVを読み込む」ボタン** を押す — フレーム数や座標範囲が表示される
3. **出力先フォルダ・データセット名** を設定する（任意）
4. **表示設定** — カラーマップ、y軸反転、時間軸の形式を選択する

#### タブ② ROI設定

ROIは最大30件まで登録でき、それぞれ独立した範囲を持ちます。各ROIは「分析に含める」チェックボックスで有効/無効を切替できます。

**ROIの新規追加（ヒートマップから選択）**

1. 代表フレームを選ぶ（ROI設定のベースとなるフレーム）
2. ヒートマップ上で **ドラッグして範囲を選択** する（ボックス選択）
3. 下部に変換後のデータ座標（x, y範囲）が表示される
4. ROI名を入力する（デフォルト: `roi_1`, `roi_2`, ...）
5. **「ROIを追加」ボタン** を押す
6. ヒートマップにROI枠がオーバーレイ表示される
7. 手順 2〜6 を繰り返し、必要な数だけ登録する

**ROIの新規追加（手動入力）**

1. 左側の「手動入力」セクションを展開する
2. ROI名、x_min、x_max、y_min、y_max を直接入力する
3. **「手動でROIを追加」ボタン** を押す

**ROIの編集**

1. 右側の ROI一覧 で、編集したい ROI の expander（折りたたみ）を開く
2. 名前・x_min・x_max・y_min・y_max を変更する
3. **「保存」ボタン** を押す（バリデーション: 名前重複チェック、座標範囲チェック）

**ROIの削除**

- ROI一覧の各 expander 内にある **「削除」ボタン** を押す

**JSON インポート / エクスポート**

- 既存の `rois.json` をアップロードして一括読み込み可能
- 現在のROI定義を JSON としてダウンロード可能

#### タブ③ 分析実行

1. タブ①でCSV読み込み済み、タブ②でROI設定済みであることを確認
2. **「▶ 分析を実行」ボタン** を押す
3. ROIごとの時系列統計（mean, std, variance）のグラフが表示される
4. **基準ROIとの差分グラフ** — 任意のROIを基準に選択し、他ROIとの差分を表示
5. ROIオーバーレイ画像・ROI切り出し画像が表示される
6. CSV / PNG / ZIP でダウンロード可能
7. **「分析結果をプロジェクトに保存」** で結果を保存し、タブ⑤からいつでも呼び出し可能

#### タブ④ ヒートマップ可視化

1. フレーム間引き数を設定（1=全フレーム、2=1フレームおき、...）
2. GIF生成の有無を選択
3. **「▶ ヒートマップを生成」ボタン** を押す
4. サマリー画像・GIFアニメーション・個別フレームブラウザが表示される
5. 各画像をダウンロード可能

### 出力先

`出力先フォルダ/データセット名/YYYYMMDD_HHmmSS/` に自動保存されます。  
例: `outputs/experiment_A/20260403_153000/`

### 設定の永続化

以下の設定はアプリ終了後も `settings.ini` に自動保存され、次回起動時に復元されます。

- カラーマップ、Y軸反転、時間軸形式
- 出力先フォルダ
- 入力ディレクトリ履歴
- 切り出しオフセット（左・右）