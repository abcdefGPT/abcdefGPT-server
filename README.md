### 실행 순서
1. 가상환경 활성화

source venv/server/Scripts/activate

2. 5000번 포트로 서버 실행

uvicorn main:app --reload --host 127.0.0.1 --port 5000