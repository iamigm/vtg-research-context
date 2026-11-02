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
