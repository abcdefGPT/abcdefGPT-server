import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import openai
# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Set API key from environment variable
GPT_API_KEY = os.getenv("GPT_API_KEY")
os.environ["OPENAI_API_KEY"] = GPT_API_KEY

# Decomposition Template
template = (
    "You need to get multiple single questions by performing decomposition to make it easier to search for multiple documents from complex questions."
    "The conditions are as follows. Please make sure that all conditions are met."
    "1. The {Query} I gave you means a complex query, and the result of decomposing Q is a single query (SQ). There can be multiple SQs (SQ1, SQ2, SQ3...)"
    "2. Decomposition should be as consistent as possible (the same complex question should be given)"
    "3. The decomposed SQ should be generated based on the common main entity in Q."
    "Examples of satisfying all conditions are as follows."
    "Q: Who is older, Annie Morton or Terry Richardson? SQ1: How old is Annie Morton? SQ2: How old is Terry? SQ1: How old is Annie Morton? SQ2: How old is Terry Richardson?"
    "Q: Was there no change in the portrayal of Google's influence on the digital ecosystem between the report from The Verge on Google's impact on the internet's appearance published on November 1, 2023, and the report from TechCrunch on a class action antitrust suit against Google published later? SQ1: What is the portrayal of Google's influence on the digital ecosystem in the report published by The Verge on Google's impact on the internet's appearance on November 1, 2023? SQ2: What is the portrayal of Google's influence on the digital ecosystem in the report published by TechCrunch on a class action antitrust suit against Google after November 1, 2023?"
    "4. For example, if you have the following Q and SQs, SQ3 is not the SQ for retrieving the document. Since these SQs must be handled by llm, it is appropriate not to be the result of query decomposition."
    "Q: Does 'The New York Times' article suggest that Connor Bedard has the potential to dominate in the NHL, while the 'Sporting News' article indicates that the USC basketball team has the potential to become a National Championship contender, or do both articles suggest a similar potential for their respective subjects? SQ1: What does 'The New York Times' article say about Connor Bedard's potential to dominate in the NHL? SQ2: What does the 'Sporting News' article say about the USC basketball team's potential to become a National Championship contender? SQ3: Do both articles suggest a similar potential for their respective subjects?"
    "5. SQs should have a high similarity to Q."
    "6. Give me Q and SQs only JSON form, and don't use another annotation. I dont need answer about SQs"
)

llm = ChatOpenAI(
    temperature=1.0,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name='gpt-4o',  # 모델명
)

prompt1 = PromptTemplate(template=template, input_variables=['Query'])
llm_chain = prompt1 | llm

async def create_vector_store(app):
    loader = DirectoryLoader(
        'rule/public_institution_rule_english',
        glob="*.json",
        show_progress=True,
        loader_cls=lambda x: TextLoader(x, encoding='utf-8')
    )
    docs = loader.load()
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings(api_key=openai.api_key))
    app.state.vectorstore = vectorstore