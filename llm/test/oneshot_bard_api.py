from bard_api import retriever

# 조회할 질의 입력
input_query = '프랑스의 수도는 어디인가요?'

# Bard-API를 사용하여 질의에 대한 정보 검색
result = retriever.retrieve(input_query)

print(result)
# 출력: "프랑스의 수도는 파리입니다."