import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import json

def user_chat(query, vectorstore, llm, llm_chain):

    QUERY = "Q: " + query
    Answer = llm_chain.invoke(QUERY)
    data = Answer.content

    modified_data = re.search(r'\{.*?\}', data, re.DOTALL)
    result = modified_data.group(0)
    data = json.loads(result)

    SQ_keys = list(data.keys())[1:]

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt") + "Please let me know the basis of your answer when you give me the final answer. (ex: Based on article and paraphraph)"

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    def process_key(key):
        return rag_chain.invoke(data[key])

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_key, key): key for key in SQ_keys}
        SQ_ans = []
        for future in as_completed(futures):
            SQ_ans.append(future.result())

    recomposition_Prompt = (
        "{Input} contains the original questions and the basis for answering them. Please combine them and create an answer.")
    prompt2 = PromptTemplate(template=recomposition_Prompt, input_variables=['Input'])
    llm_chain = prompt2 | llm
    tmp = [QUERY] + SQ_ans
    combined_prompt = "\n".join(tmp) + "\n"
    final_Answer = llm_chain.invoke(combined_prompt)

    return final_Answer.content