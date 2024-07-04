# Hogangnono Chrome-Extension 

광운대학교 산학협력캡스톤 **abcdefGPT TEAM Project**

# abcdefGPT
<img src="https://avatars.githubusercontent.com/u/154737156?s=200&v=4" width="200" height="200">


## 프로젝트 소개
생성형 AI기반 업무 지원 어시스턴트 서비스를 제공하기 위한 프로젝트입니다.

기존 방식보다 더 효과적인 쿼리 분해 방법(Query Decomposition)을 적용하여 	
분할 Reranking을 통해 다중 문서 환경에서도 높은 성능을 발휘하는 RAG 개발

### Architecture

<img width="954" alt="스크린샷 2024-06-08 오후 8 17 13" src="https://github.com/abcdefGPT/abcdefGPT-FE/assets/92677088/1a733571-efbf-46c3-85f0-16ff91966af7">

### 동작 과정
<img width="371" alt="스크린샷 2024-06-08 오후 8 18 03" src="https://github.com/abcdefGPT/abcdefGPT-FE/assets/92677088/46ad2c80-3148-4f4d-9b31-154b0cd26e55">

## 기능
![편집-화면](https://github.com/abcdefGPT/abcdefGPT-FE/assets/92677088/9da87bd3-c08c-4756-b7e1-d96d6fb18a66)
<!-- - 기능 1 -->
### 진행상황
**API 연결 및 순차적으로 글자 출력 진행**
- 현재 기능 추가 개선 진행 중

## 팀원 소개
>  **FE 1명 & BE 1명 & 인공지능모델 3명**
> 
> 모델 개발에 필요한 데이터 라벨링은 공동 진행


<table>
  <tr>
    <td><img src="https://github.com/Mulsanne2.png" width="100px" /></td>
    <td><img src="https://github.com/JungSungYeob.png" width="100px" /></td>
    <td><img src="https://github.com/bbang-jun.png" width="100px" /></td>
    <td><img src="https://github.com/Bjimin.png" width="100px" /></td>
    <td><img src="https://github.com/syoooooung.png" width="100px" /></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Mulsanne2">권민철</a>
    </td>
    <td align="center"><a href="https://github.com/JungSungYeob">정성엽</a>
    </td>
    <td align="center"><a href="https://github.com/bbang-jun">강병준</a>
    </td>
    <td align="center"><a href="https://github.com/Bjimin">방지민</a>
    </td>
    <td align="center"><a href="https://github.com/syoooooung">황세영</a>
    </td>

  </tr>
  <tr>
    <td align="center">AI Model
    </td>
    <td align="center">Frontend
    </td>
    <td align="center">Backend
    </td>
    <td align="center">AI Model
    </td>
    <td align="center">AI Model
    </td>
  </tr>
</table>

**컴파일 방법**
- 실시간 컴파일
  ```
  npm start
  ```

- 컴파일
  ```
  npm run build
  ```
  


### 실행 순서
1. 가상환경 활성화

source venv/server/Scripts/activate

2. 5000번 포트로 서버 실행

uvicorn main:app --reload --host 127.0.0.1 --port 5000