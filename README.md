<div align="center">
  <h1>AIRO Flatness</h1>
  <p><b>3D 포인트 클라우드 기반 바닥 평탄도 분석 시스템</b></p>
  <p>LiDAR 스캔 데이터(PLY)를 로딩하여 바닥면을 자동 추출하고,<br>평탄도 분석 및 시각화 리포트를 생성합니다.</p>
</div>

---

## 주요 기능

- **대용량 PLY 스트리밍 로딩** — 27GB+ 파일도 청크 단위 스트리밍 + 랜덤 샘플링으로 메모리 효율적 처리
- **GPU 복셀 다운샘플링 + NPZ 캐시** — CuPy 기반 복셀 그리드 다운샘플링, 결과를 `data/downsample/`에 캐싱하여 재실행 시 로딩 생략
- **3단계 하이브리드 바닥 추출** — Z-히스토그램 피크 감지 → Z-필터링 → Intensity/Color 정제
- **바닥 평탄도 분석** — 그리드 기반 SVD 표면 법선 추정으로 셀별 기울기 계산
- **GPU 가속 3D 시각화** — PyVista 기반 4가지 뷰 모드 + 5방향 스크린샷 캡처
- **7종 분석 차트 자동 생성** — Z-히스토그램, 필터링 퍼널, 평탄도 히트맵 등
- **JSON 리포트** — 분석 메타데이터, 파라미터 민감도, 평탄도 통계 포함
- **Figure Tool** — 관심영역(ROI)을 마우스로 선택해 높이 프로파일 + 3D 표면 메시를 생성하는 별도 진입점

---

## 분석 결과 예시

### 3D 포인트 클라우드 시각화

> 바닥 영역이 빨간색으로 하이라이트된 전방 뷰 (Mode 4: Highlighted Floor)

| Before | After |
|:---:|:---:|
| ![Before Highlighted](docs/images/before_highlighted_front.png) | ![After Highlighted](docs/images/after_highlighted_front.png) |

<details>
<summary>전체 포인트 클라우드 (Mode 1: Full Point Cloud)</summary>

![Full Point Cloud](docs/images/before_full_front.png)

</details>

### 바닥 추출 분석

#### Z-히스토그램 피크 감지

> Z축 분포에서 바닥면 피크를 자동 감지하고, FWHM 기반으로 바닥 범위를 결정합니다.

| Before | After |
|:---:|:---:|
| ![Before Z-Histogram](docs/images/before_z_histogram.png) | ![After Z-Histogram](docs/images/after_z_histogram.png) |

#### 3단계 필터링 퍼널

> 전체 포인트 → Z-필터 → Intensity/Color 정제를 거쳐 최종 바닥 포인트를 추출합니다.

![Filtering Funnel](docs/images/before_filtering_funnel.png)

#### 바닥 비율

| Before (34.5%) | After (48.4%) |
|:---:|:---:|
| ![Before Floor Ratio](docs/images/before_floor_ratio.png) | ![After Floor Ratio](docs/images/after_floor_ratio.png) |

### 평탄도 분석

> 바닥 포인트를 그리드로 분할하여 각 셀의 기울기(Tilt)를 SVD 표면 법선으로 계산합니다.

| Before (평균 기울기 10.5°) | After (평균 기울기 16.0°) |
|:---:|:---:|
| ![Before Flatness](docs/images/before_flatness_heatmap.png) | ![After Flatness](docs/images/after_flatness_heatmap.png) |

### 파라미터 민감도 분석

> width_multiplier, color_tolerance, intensity_percentile 3개 파라미터 변화에 따른 바닥 비율 변화를 분석합니다.

![Parameter Sensitivity](docs/images/before_parameter_sensitivity.png)

---

## 프로젝트 구조

```
airo-fitness/
├── src/
│   ├── main.py                  # 메인 진입점 (소스 선택 → 로딩 → 바닥 추출 → 차트 → 3D 뷰어)
│   ├── figure_tool.py           # Figure Tool 진입점 (ROI 선택 → 높이 프로파일 → 3D 표면 메시)
│   ├── config.py                # 중앙 설정 (Config 데이터클래스)
│   ├── utils.py                 # 대화형 파일/소스 선택 프롬프트, 진행률 표시, 서브샘플링
│   ├── loader/
│   │   └── ply_loader.py        # PLY 파일 스트리밍 로더 + 랜덤 샘플링
│   ├── preprocessing/
│   │   ├── pipeline.py          # 로딩 → 다운샘플링 → 캐시 저장/로드 오케스트레이터
│   │   ├── downsampling.py      # CuPy 복셀 그리드 다운샘플링 (청크 + 트리 머지)
│   │   └── cache.py             # NPZ 캐시 읽기/쓰기, 캐시 경로 생성
│   ├── extractor/
│   │   ├── peak_detector.py     # Z-히스토그램 피크 감지 (scipy.signal)
│   │   ├── floor_extractor.py   # 3단계 바닥 추출 파이프라인
│   │   └── flatness_analyzer.py # 그리드 기반 SVD 평탄도 분석
│   ├── figure/
│   │   ├── detrend.py           # 평면 피팅 + Z 디트렌드 (GPU/CPU 자동 분기)
│   │   ├── roi_selector.py      # XY 폴리곤 ROI 선택기 + Z 범위 히스토그램 선택기
│   │   ├── height_profile.py    # X축 방향 높이 프로파일 (Z 잔차 기반)
│   │   ├── surface_3d.py        # Delaunay 삼각분할 → PyVista 표면 메시
│   │   └── roi_context.py       # ROI 오버레이 2D 탑뷰 / 3D 컨텍스트 렌더
│   ├── viewer/
│   │   └── visualizer.py        # PyVista 3D 뷰어 (4 모드 + 스크린샷)
│   └── chart/
│       ├── chart_manager.py     # 7개 차트 + JSON 리포트 오케스트레이터
│       ├── histogram_charts.py  # Z-히스토그램, Intensity, 색상 거리 차트
│       ├── summary_charts.py    # 필터링 퍼널, 바닥 비율 도넛 차트
│       ├── parameter_sensitivity.py # 파라미터 민감도 라인 차트
│       ├── flatness_heatmap.py  # 평탄도 히트맵
│       └── report_writer.py     # JSON 리포트 생성기
├── tests/                       # pytest 테스트 (pythonpath = ["src"])
├── data/                        # PLY 입력 파일 + downsample/ 캐시 (git 미추적)
├── results/                     # 분석 결과 출력 (git 미추적)
├── docs/images/                 # README 에셋 이미지
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.14+ |
| 수치 연산 | NumPy 2.4+ |
| 신호 처리 | SciPy 1.17+ (피크 감지, SVD) |
| GPU 연산 | CuPy (cupy-cuda12x 13+) — 미설치/미동작 시 NumPy로 자동 폴백 |
| 3D 시각화 | PyVista 0.47+ (VTK 기반, GPU 가속) |
| 차트 | Matplotlib 3.10+ |
| 테스트 | pytest 9+ |
| 패키지 관리 | uv |
| 컨테이너 | Docker (CPU / GPU) |

---

## 설치 및 실행

### 사전 요구사항

- Python 3.14 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

### 로컬 실행

```bash
# 의존성 설치
uv sync

# 실행 (data/ 디렉토리에 PLY 파일 필요)
uv run python src/main.py

# Figure Tool (ROI 선택 → 높이 프로파일 + 3D 표면 메시)
uv run python src/figure_tool.py

# 테스트
uv run pytest
```

### Docker 실행

```bash
# CPU 모드
docker compose up -d airo-fitness
docker compose exec airo-fitness bash
uv sync && uv run python src/main.py

# GPU 모드 (NVIDIA GPU 필요)
docker compose --profile gpu up -d airo-fitness-gpu
docker compose exec airo-fitness-gpu bash
uv sync && uv run python src/main.py
```

> 3D 뷰어는 `$DISPLAY`가 필요합니다. Docker에서는 호스트의 X11 포워딩이 활성화되어 있어야 하며, compose 기본값은 `host.docker.internal:0`입니다.

---

## 실행 흐름

```
1. 소스 선택     원본 PLY / 다운샘플 NPZ 캐시 중 선택
       ↓
2. 스트리밍 로딩  청크 단위 읽기 + 랜덤 샘플링 (기본 500만 포인트)
       ↓
3. 다운샘플링    GPU 복셀 그리드 다운샘플 → data/downsample/ 에 NPZ 캐시
       ↓
4. 바닥 추출     Z-히스토그램 피크 → Z-필터 → Intensity/Color 정제
       ↓
5. 차트 생성     7종 분석 차트 + JSON 리포트 → results/{timestamp}/
       ↓
6. 3D 시각화    PyVista GPU 렌더링 (키보드 인터랙션)
```

캐시 유효성은 원본 PLY의 `mtime`과 `size`가 모두 일치하는지로 판단합니다. 2단계에서 선택한 소스가 캐시라면 1~3단계를 건너뜁니다.

### Figure Tool 흐름

```
소스 선택 → 로딩/다운샘플 → XY ROI 선택(4점 클릭) → Z 범위 선택(히스토그램 드래그)
   → 높이 프로파일 PNG → 3D 표면 메시 → ROI 컨텍스트 뷰(2D/3D)
   → results/{파일명}_figure_{timestamp}/ 에 저장
```

---

## 뷰어 조작법

| 키 | 동작 |
|----|------|
| `1` | Full Point Cloud — 전체 포인트 표시 |
| `2` | Floor Only — 바닥 포인트만 표시 |
| `3` | Non-Floor Only — 비바닥 포인트만 표시 |
| `4` | Highlighted Floor — 바닥을 빨간색으로 하이라이트 (기본) |
| `S` | 현재 모드의 5방향 스크린샷 캡처 (top, front, back, right, left) |

---

## 설정 파라미터

`src/config.py`의 `Config` 데이터클래스에서 모든 하이퍼파라미터를 관리합니다.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `max_points` | 5,000,000 | 최대 샘플링 포인트 수 |
| `chunk_size` | 1,000,000 | 스트리밍 청크 크기 |
| `downsampling_voxel_size` | 0.0001 | 복셀 다운샘플링 격자 크기(m) — 작을수록 포인트 증가 |
| `num_bins` | 200 | Z-히스토그램 빈 수 |
| `width_multiplier` | 2.5 | FWHM 기반 바닥 범위 배율 |
| `intensity_percentile` | 25.0 | Intensity 필터 퍼센타일 임계값 |
| `color_tolerance` | 0.6 | 색상 거리 허용치 |
| `flatness_target_grid` | 150 | 평탄도 분석 그리드 크기 |
| `point_size` | 2.0 | 3D 뷰어 포인트 렌더링 크기 |
| `chart_dpi` | 150 | 차트 이미지 해상도 |
| `fig_heatmap_target_grid` | 100 | Figure Tool 높이 프로파일 격자 크기 |
| `fig_z_exaggeration` | 1.0 | 3D 표면 메시 수직 배율 |
| `fig_dpi` | 300 | Figure Tool 이미지 해상도 |

---

## 분석 파이프라인 상세

### 1단계: Z-히스토그램 피크 감지

Z축 값의 히스토그램에서 `scipy.signal.find_peaks`로 바닥면 피크를 자동 감지합니다. FWHM(반치전폭)을 계산하여 바닥 Z 범위를 결정합니다. 기울어진 바닥의 경우 틸트 보정이 적용됩니다.

### 2단계: Z-범위 필터링

감지된 피크의 Z 범위(`peak_z ± FWHM × width_multiplier`)에 해당하는 포인트를 1차 필터링합니다.

### 3단계: Intensity/Color 정제

- **Intensity 필터**: 바닥 포인트의 intensity 분포에서 하위 퍼센타일 이하 제거
- **Color 필터**: 바닥 포인트의 중앙 색상 대비 거리가 임계값 초과인 포인트 제거

### 평탄도 분석

바닥 포인트를 X-Y 그리드로 분할한 후, 각 셀에서 SVD(특이값 분해)로 표면 법선 벡터를 추정합니다. 법선 벡터와 Z축 사이의 각도가 해당 셀의 기울기(Tilt)입니다.

---

## 출력 결과물

`results/{timestamp}/` 디렉토리에 다음 파일들이 생성됩니다:

| 파일 | 설명 |
|------|------|
| `01_z_histogram_peak.png` | Z-히스토그램 + 피크 오버레이 |
| `02_filtering_funnel.png` | 3단계 필터링 퍼널 차트 |
| `03_intensity_histogram.png` | Intensity 분포 히스토그램 |
| `04_color_distance.png` | 색상 거리 히스토그램 |
| `05_floor_ratio.png` | 바닥/비바닥 비율 도넛 차트 |
| `06_parameter_sensitivity.png` | 파라미터 민감도 분석 |
| `07_flatness_heatmap.png` | 평탄도 히트맵 |
| `report.json` | 분석 메타데이터 + 통계 리포트 |

Figure Tool은 별도 디렉토리 `results/{파일명}_figure_{timestamp}/` 에 `height_profile.png`와 ROI 컨텍스트 캡처를 저장합니다.
