# obsidian-rag
Obsidian-Rag는 Langchain을 활용하여 마크다운 파일에서 RAG를 수행하는 로컬 우선 프로젝트입니다. 특히 우리는 모두 열혈 팬들이라는 것을 알기 때문에(워털루 강해요 💪) Obsidian 노트 작성 앱과 함께 사용하도록 설계되었습니다.

## 기능
- 주어진 디렉토리에서 마크다운 파일 로드.
- 로드된 파일을 벡터화하여 추가 처리.
- 벡터화된 데이터에서 유사도 검색 수행.
- `ChatOllama`, `ObsidianLoader`, `OllamaEmbeddings`, `Chroma`와 같은 Langchain 라이브러리의 기능 활용.

## 의존성
`requirements.txt`를 설치해야 하며 `Mistral`을 사용하는 `Ollama` 인스턴스가 실행 중이어야 합니다.

### 사용법

프로젝트의 메인 스크립트는 obsidian_rag.py입니다. 파일 경로와 파일을 벡터화할지 여부를 결정하는 불리언 플래그를 명령줄 인수로 받습니다.

`python obsidian_rag.py --filepath YOUR_FILE_PATH --vectorize`

이 명령은 아직 작업 중인 그라디오 인터페이스를 열며, 이를 채팅 인터페이스로 만들어야 합니다.
<img width="688" alt="image" src="https://github.com/ParthSareen/obsidian-rag/assets/29360864/13747e0b-78f8-495e-9f03-c80229d537a6">
<img width="1256" alt="image" src="https://github.com/ParthSareen/obsidian-rag/assets/29360864/f79e90e3-2624-46a9-90e0-12034c9afb42">