# Obsidian RAG

![](https://i.imgur.com/JyjTmcG.png)

Obsidian-Rag는 Langchain을 활용하여 마크다운 파일에서 RAG를 수행하는 로컬 우선 프로젝트입니다. Obsidian 노트 작성 앱과 함께 사용하도록 설계되었습니다.

## 기능

- 주어진 디렉토리에서 마크다운 파일 로드.
- 로드된 파일을 벡터화하여 추가 처리.
- 벡터화된 데이터에서 유사도 검색 수행.
- `ChatOllama`, `ObsidianLoader`, `OllamaEmbeddings`, `Chroma`와 같은 Langchain 라이브러리의 기능 활용.

## 의존성

`poerty`를 사용하여 의존성 패키지를 설치해야 합니다.

```sh
poetry install
```

### 사용법

프로젝트의 메인 스크립트는 main.py입니다.

```sh
poetry run streamlit run main.py
```

