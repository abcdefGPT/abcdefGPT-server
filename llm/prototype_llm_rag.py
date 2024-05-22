import re

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import os
import json
import openai
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from dotenv import load_dotenv

load_dotenv()
GPT_API_KEY = os.getenv("GPT_API_KEY")

os.environ["OPENAI_API_KEY"]=GPT_API_KEY

llm = ChatOpenAI(temperature=0,               # 창의성 (0.0 ~ 2.0)
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-4o',  # 모델명
                )

#Decomposition Template
template = ("You need to get multiple single questions by performing decomposition to make it easier to search for multiple documents from complex questions."+
            "The conditions are as follows. Please make sure that all conditions are met."+
            "1. The {Query} I gave you means a complex query, and the result of decomposing Q is a single query (SQ). There can be multiple SQs (SQ1, SQ2, SQ3...)"+
            "2. Decomposition should be as consistent as possible (the same complex question should be given)"+
            "3. The decomposed SQ should be generated based on the common main entity in Q."+
            "Examples of satisfying all conditions are as follows."+
            "Q: Who is older, Annie Morton or Terry Richardson? SQ1: How old is Annie Morton? SQ2: How old is Terry? SQ1: How old is Annie Morton? SQ2: How old is Terry Richardson?"+
            "Q: Was there no change in the portrayal of Google's influence on the digital ecosystem between the report from The Verge on Google's impact on the internet's appearance published on November 1, 2023, and the report from TechCrunch on a class action antitrust suit against Google published later? SQ1: What is the portrayal of Google's influence on the digital ecosystem in the report published by The Verge on Google's impact on the internet's appearance on November 1, 2023? SQ2: What is the portrayal of Google's influence on the digital ecosystem in the report published by TechCrunch on a class action antitrust suit against Google after November 1, 2023?"+
            "4. For example, if you have the following Q and SQs, SQ3 is not the SQ for retrieving the document. Since these SQs must be handled by llm, it is appropriate not to be the result of query decomposition."+
            "Q: Does 'The New York Times' article suggest that Connor Bedard has the potential to dominate in the NHL, while the 'Sporting News' article indicates that the USC basketball team has the potential to become a National Championship contender, or do both articles suggest a similar potential for their respective subjects? SQ1: What does 'The New York Times' article say about Connor Bedard's potential to dominate in the NHL? SQ2: What does the 'Sporting News' article say about the USC basketball team's potential to become a National Championship contender? SQ3: Do both articles suggest a similar potential for their respective subjects?"+
            "5. SQs should have a high similarity to Q."
            "6. Give me Q and SQs only JSON form, and don't use another annotation. I dont need answer about SQs")
# 결합
prompt1 = PromptTemplate(template=template, input_variables=['Query'])
# 질의내용
llm_chain = prompt1|llm
#Query 받기
QUERY = input("Enter your Question:")
QUERY = "Q: "+QUERY
print(QUERY)

Answer = llm_chain.invoke(QUERY)
print(Answer.content)
data = Answer.content
#JSON 변환 과정
modified_data = re.search(r'\{.*?\}',data,re.DOTALL)
result = modified_data.group(0)
data = json.loads(result)

print(data)

SQ_keys = list(data.keys())[1:]

for key in SQ_keys:
    print(f"{key}: {data[key]}")

#JSON 로드
loader = DirectoryLoader('../rule/public_institution_rule_english/', glob="*.json", show_progress=True, loader_cls=TextLoader)

docs = loader.load()

print(f'document count : {len(docs)}')
print(docs[0])

vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings(api_key=openai.api_key))

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

SQ_ans = []
#답 찾기
for key in SQ_keys:
    print("-----------------------")
    print(f"{key}: {data[key]}")
    print("Answer")
    SQ_Answer = rag_chain.invoke(data[key])
    SQ_ans.append(SQ_Answer)
    print(SQ_Answer)

print("--------SQ_ans-------")
for i, s in enumerate(SQ_ans, 1):
    print(f"{i}. {s}")
#print(data['SQ1'])

#Recomposition prompt
recomposition_Prompt = ("{Input} contains the original questions and the basis for answering them. Please combine them and create an answer.")

prompt2 = PromptTemplate(template=recomposition_Prompt, input_variables=['Input'])
llm_chain = prompt2|llm
tmp = [QUERY]+SQ_ans
combined_prompt = "\n".join(tmp)+"\n"
print("------combined------")
print(combined_prompt)
print("------Final Answer--------")
final_Answer = llm_chain.invoke(combined_prompt)
print(final_Answer.content)