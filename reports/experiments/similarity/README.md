# Frame–query similarity

고정 pilot 300쌍(Charades·ActivityNet·QVHighlights 각 100개)에 대한 CLIP ViT-L/14 및 Qwen3-VL-Embedding-8B의 독립 frame/query cosine 비교.

- [정량 보고서](results/pilot_300_v1/REPORT.md)
- [샘플별 HTML](results/pilot_300_v1/report.html)
- [모델 다운로드 및 weight SHA256](../../backbones/manifests/similarity_download.json)

## 전체 9,404쌍 실험

Charades 3,363쌍, ActivityNet 4,500쌍, QVHighlights 1,541쌍(고유 영상 4,279개)의 전체 manifest를 같은 설정으로 처리한다. Pilot의 호환 frame·영상 embedding은 hard link로 재사용한다. 결과는 `results/full_9404_v1/`에 저장하며 완료 후 MD·영상 재생 HTML·최신 PNG와 검증 결과를 자동 생성한다.

- [통합 진행 로그](artifacts/runs/full_9404_v1/logs/pipeline.log)
- [기계 판독용 상태](artifacts/runs/full_9404_v1/logs/status.json)
- 모델별 로그: `artifacts/runs/full_9404_v1/logs/clip.log`, `qwen.log`. 후처리·검증: `finalize.log`.

실행·재개:

```bash
SIMILARITY_RUN=full_9404_v1 \
SIMILARITY_MANIFEST=/home/irteam/experiments/zero-shot/artifacts/common/predictions/legacy/qwen3/samples.json \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home/irteam/experiments/_shared/timelens/.venv/bin/python -u _lib/run_pipeline.py \
  >> artifacts/runs/full_9404_v1/logs/pipeline.log 2>&1
```

중복 실행은 lock으로 차단한다. 통합 로그는 20초마다 모델별 완료 영상 수·비율, dataset별 embedding 준비 query 수, 경과 시간과 embedding 단계의 대략적인 ETA를 기록한다. 실패하면 관련 프로세스를 종료하고 실패 원인과 개별 로그 경로를 남긴다. 인자 없이 기존 개별 스크립트를 실행하면 pilot 경로를 사용한다.

공유 환경: `/home/irteam/experiments/_shared/timelens/.venv/bin/python`.
실행 코드는 `_lib/`, 공식 Qwen 구현은 `_lib/vendor/Qwen3-VL-Embedding/`에 보관한다. 모델은 `/home/irteam/backbones/models/`에 revision을 고정해서 저장한다.

원본 영상에서 2fps로 선택한 같은 frame을 두 모델에 제공하고 FP32 L2-normalized embedding의 내적을 계산한다. 모델별 native image preprocessing은 유지한다. GT는 `[start,end)` membership으로 정의한다. Query별 AP → macro mAP, 첫 GT frame 순위·MRR, GT mass·uniform baseline·lift를 측정한다. Softmax temperature는 .01/.03/.07/.1/.3/1.0이고 .07은 시각화 기본값이다.

재실행 방법과 상세 설정은 보고서에 기록한다. `artifacts/runs/<run>/logs/`는 원본 실행 기록이며 batch 증가를 위한 중단·재시작도 포함한다. 완료한 영상의 embedding은 재사용한다.

보고서와 데이터는 같은 실행 이름으로 연결한다.

```text
results/<run>/              # REPORT.md, report.html, viewer_desktop.png, viewer_mobile.png
artifacts/runs/<run>/       # clip/, qwen/, frames/, media/, 수치·설정·검증
artifacts/runs/<run>/logs/  # 실행·후처리 로그
artifacts/common/          # 실행 간 공유하는 출처
```

HTML은 `../../artifacts/runs/<run>/`의 이미지·영상을 상대경로로 읽는다. 영상 재생, GT 시작점 이동, 그래프·frame 클릭 시 seek와 재생 위치 커서를 지원한다. 공유 시 프로젝트의 `results/`와 `artifacts/` 관계를 함께 유지해야 한다.

Frame timestamp·원본 식별 metadata, embedding, FP32 score와 실행 로그는 보존한다. 최신 preview는 데스크톱·모바일 각각 같은 이름으로 덮어쓴다. 임시 preview·검증 파일은 누적하지 않는다. 저장 기준은 [상위 안내](../README.md)를 따른다.

추론 처리량: 여러 영상의 frame을 합쳐 CLIP 256 / Qwen 128 batch를 채우고 CPU DataLoader prefetch로 전처리와 GPU 추론을 겹친다. 현재 CPU quota 12코어에 맞춰 영상 준비 worker 6개, CLIP 전처리 1개, Qwen 전처리 3개를 사용한다. 로그의 `PREFETCH` 줄에서 batch 크기·CPU 대기·GPU 처리 시간·처리량을 확인할 수 있다. `infer.py --workers 0`은 기존 영상별 동기 방식이다. Pilot은 300 query–video 쌍, 고유 영상 292개이므로 embedding 캐시 재사용 개수와 평가 샘플 개수를 구분한다.

## Pilot 300 · 16fps 비용 비교

실행 ID: `pilot_300_16fps_cost_v1`. CLIP ViT-L/14와 Qwen3-VL-Embedding-8B의 frame/query cosine 및 Qwen3-VL-8B-Instruct의 video→query greedy answer generation을 같은 16fps source frame으로 비교한다. 영상 전체를 사용하며 768-frame cap을 비활성화한다. 준비→CLIP→Qwen embedding→Instruct→보고서·검증을 순차 실행하고, 모델 loading·warmup을 측정에서 제외한다.

- [진행 로그](artifacts/runs/pilot_300_16fps_cost_v1/logs/benchmark.log)
- [현재 상태](artifacts/runs/pilot_300_16fps_cost_v1/logs/status.json)
- 완료 보고서·HTML·최신 PNG: `results/pilot_300_16fps_cost_v1/`.

보고서는 샘플별 GPU forward·CUDA event·전처리 포함 wall time, 실제 stage wall time, 입력/생성 token 수, 추정 FLOPs를 포함한다. Instruct의 min_tokens64·total_tokens14336 soft spatial budget을 유지하며 실제 visual token 수를 기록한다. 다른 GPU 작업이 감지되면 timing 오염을 막기 위해 중단한다. 저장 및 재개 방법은 [artifacts 안내](artifacts/README.md)를 따른다.

인접 frame cosine은 별도 실험 없이 기존 2fps·16fps pilot과 full 결과에 통합한다. 각 뷰어의 인접 frame 그래프는 query와 무관한 `cos(f_(i-1), f_i)`를 나중 timestamp에 표시하며, 클릭하면 두 frame을 함께 보여준다. 기존 frame–query 성능 및 비용 측정값은 유지한다.
