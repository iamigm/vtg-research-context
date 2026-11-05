# Markdown math renderer

공통 family viewer는 로컬 KaTeX로 inline `$...$`·`\(...\)`와 독립된 `$$` block을 표시한다. Markdown 원문의 수식이 원본이다. 외부 CDN 요청 없이 `katex/katex.min.js`, `katex/katex.min.css`, `katex/fonts/`를 사용한다.

공식 npm 배포 archive의 integrity를 검증한 뒤 필요한 dist 파일과 MIT LICENSE만 보존했다. 버전·archive URL·integrity·각 파일 SHA-256은 [manifest](katex/manifest.json)에 있다. 설정은 `trust: false`, `throwOnError: false`, `maxExpand: 1000`이며 일반 HTML은 escape한다. `<details>`·`<summary>`는 공통 Markdown parser가 허용한 최소 속성만 처리한다.

표시 코드는 [family_view.js](../../_lib/family_view.js), 스타일은 [family_view.css](../../_lib/family_view.css)에 있다. 공식 문서는 [KaTeX browser installation](https://katex.org/docs/browser)과 [options](https://katex.org/docs/options.html)을 따른다.
