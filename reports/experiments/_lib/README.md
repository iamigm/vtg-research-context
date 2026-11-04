# 실험 디렉터리 관리 도구

결과 경로 변환, 정리 검증, 뷰어 검증을 위한 공통 코드다. 가상환경과 외부 의존성은 [별도 공통 자원](../_shared/README.md)에 둔다. 실험별 구현은 각 실험의 `_lib/`에 둔다.

- `result_layout.py`: 기존 경로를 정리된 실제 위치로 해석
- `validate_result_layout.py`: 이동 전 수치·파일 기록과 현재 링크·구문 검증
- `check_result_viewers.py`: 뷰어와 영상 재생 확인, 최신 preview 덮어쓰기
- `finalize_project_roots.py`: 실행 중인 작업 종료 후 임시 경로 링크 제거와 자동 검증

최초 이동 스크립트는 재현 기록이며 중복 적용을 차단한다. 현재 정리 현황은 [상위 안내](../README.md)에 기록한다.

- `build_index.py`: 전체 실험의 단일 `index.html` 생성과 복귀 링크 갱신
- `serve_index.py`: 통합 목록·실험 페이지·영상 재생 제공 (기본 포트 8780)

- `index_compare.js` · `index_compare.css`: 통합 목록의 두 실험 선택·동시 열람 UI (기존 통합 서버 사용)
- `check_comparison.py`: 프록시 경로의 선택·독립 영상 재생·교체·URL 복원·모바일 배치 검증

- `index_catalog.js` · `index_catalog.css`: 프로젝트 핸들 드래그·상단 고정·위/아래 이동·검색. 순서와 고정은 localStorage에 저장하고 같은 주소의 다른 탭에도 동기화한다. 실험 링크와 비교 시작은 새 탭을 연다.
- `experiment_theme.css`: index·비교·계열 대표 뷰·상세 HTML의 공통 글꼴, 색상, 입력 요소, 표, 헤더. `build_index.py`가 상세 HTML에 스타일 참조를 넣어 재생성 후에도 적용한다. 그래프와 정답/오답 등 데이터 색상은 각 분석 정의를 유지한다.
- `check_index_preferences.py`: 실제 포인터 드래그, 저장·복원, 상단 고정, 검색, 새 탭, 모바일, 저장 불가 시 동작, 전체 상세 페이지 테마 검증. 공용 브라우저 의존성에 맞춰 `/usr/bin/python3`로 실행한다.

- `history_controls.js`: index·계열·상세·비교 화면의 실제 브라우저 뒤로/앞으로 이동. Chrome의 이동 기록에 따라 사용 가능한 화살표만 활성화한다.
- 계열 화면 상단은 이동 화살표, `목록` 토글, `결과`/`보고서` 탭, 원본 MD와 새 탭 버튼으로 구성한다. 사이드바는 기본으로 접으며, 연결된 보고서와 결과만 같은 탭 그룹에서 전환한다. iframe 로드는 별도 이동 기록을 추가하지 않는다.
- `check_view_controls.py`: 결과/보고서 전환, 키보드 탭 조작, 페이지 안/밖 뒤로·앞으로 이동, 새 탭의 이동 기록 경계, 모바일 검증. 최신 기록은 `../artifacts/maintenance/view_controls_validation.json`이다.
