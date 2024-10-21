import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI
import os
from groq import Groq
import networkx as nx
from llm.llm_prompt import template, NER_prompt, RE_prompt, graph_to_text, relation_extraction_prompt

file_path = 'llm/vector_store/hotpotqa_test_compressed_vectorstore_ver2.index/'

# Set API key from environment variable
GPT_API_KEY = os.getenv("GPT_API_KEY")
os.environ["OPENAI_API_KEY"] = GPT_API_KEY

embeddings = OpenAIEmbeddings()

# FAISS 인덱스 파일 로드
loaded_vectorstore = FAISS.load_local(
    folder_path=file_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 1})


# Define LLM
answer_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1) # or gpt-4o-mini gpt-3.5-turbo

prompt = ChatPromptTemplate.from_template(template)

# Setup RAG pipeline
rag_chain = (
    {"context": RunnablePassthrough(),  "question": RunnablePassthrough()}
    | prompt
    | answer_llm
    | StrOutputParser()
)

# gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
model = "gpt-4o"

# entity 뽑기
def get_entities(query):
  messages = [{
      "role": "system",
      "content": NER_prompt
  }, {
      "role": "user",
      "content": query
  }]
  response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content

# re 뽑기
def get_re(words, query):
  all = "Question: " + query + "\nList of Entities:" + words
  messages = [{
      "role": "system",
      "content": RE_prompt
  }, {
      "role": "user",
      "content": all
  }]
  response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
  )

  relations = response.choices[0].message.content

  all = "Sentense: " + query + "\nEntity pairs:\n" + relations
  messages = [{
      "role": "system",
      "content": relation_extraction_prompt
  }, {
      "role": "user",
      "content": all
  }]
  response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content

def convert_rener(entities, relations):
    # 엔티티 추출
    entities = [item.strip() for item in entities.split(",")]

    # 관계 추출
    extracted_relations = []
    lines = relations.splitlines()
    for line in lines:
        parts = line.split("|")
        if len(parts) == 3:
            subject = parts[0].strip()
            relation = parts[1].strip()
            obj = parts[2].strip()
            extracted_relations.append({'subject': subject, 'relation': relation, 'object': obj})

    result = {
        'entities': entities,
        'relations': extracted_relations
    }

    return result

gclient = Groq(
    api_key=os.getenv("GROQ")
)

def process_data(prompt):

    """Send one request and retrieve model's generation."""

    chat_completion = gclient.chat.completions.create(
        messages=prompt, # input prompt to send to the model
        model="llama3-70b-8192", # according to GroqCloud labeling
        temperature=0.5, # controls diversity
        max_tokens=128, # max number tokens to generate
        top_p=1, # proportion of likelihood weighted options to consider
        stop=None, # string that signals to stop generating
        stream=False, # if set partial messages are sent
    )
    return chat_completion.choices[0].message.content

def query_decomposition(rener_json):
  decomposed_queries = []

  G = nx.DiGraph()

  # G에 entity를 node로 추가
  for nodes in rener_json['entities']:
    G.add_node(nodes)

  # rener_json 데이터에서 relations를 edge로 추가
  for edge in rener_json['relations']:
    G.add_edge(edge['subject'], edge['object'], relation=edge['relation'])

  # 서브그래프 분해를 위한 함수
  def create_subgraphs(G):
      will_decom_graph = []
      subgraphs = []

      # 연결된 성분(weakly connected components)으로 G를 분할
      # 일단 연결된 그래프별로 나눠놓기
      for component in nx.weakly_connected_components(G):
          # 각 성분별로 서브그래프 생성
          component_subgraph = G.subgraph(component).copy()
          will_decom_graph.append(component_subgraph)

      # if will_deom_graph에 요소가 있다면, 요소 하나씩(A라고 가정) 아래 코드 진행
      # 아니라면 끝내고 subgraphs 반환
      while will_decom_graph:
        # 에러 대비
        if len(will_decom_graph) > 20:
          subgraphs.append(G)
          return subgraphs

        split_by_incoming = False  # 들어오는 엣지로 분해되었는지 여부를 추적
        split_by_outgoing = False  # 나가는 엣지로 분해되었는지 여부를 추적
        # print(len(will_decom_graph)) #test
        A = will_decom_graph.pop(0)

        for node in A.nodes:
            # 나가는 edge가 2개 이상인 노드 찾기
            out_edges = list(A.out_edges(node))
            in_edges = list(A.in_edges(node))

            if len(out_edges) > 1:
              for out_edge in out_edges:
                  # print(out_edge) #test
                  subgraph = nx.DiGraph()
                  # 현재 노드와 나가는 엣지 하나 추가
                  subgraph.add_edge(node, out_edge[1], relation=G.edges[out_edge]['relation'])

                  # 나가는 엣지를 따라 생성된 서브그래프 추가

                  # subgraph에 해당 out_edge에 연결된 노드 및 그에 연결된 모든 subgraph를 추가
                  # subgraph에 해당 노드로 들어오는 edge들에 연결된 모든 subgraph를 추가

                  # 나가는 엣지에 연결된 노드 및 그에 연결된 서브그래프 추가
                  # 근데 그 나가는 엣지에 연결된 노드만 연결하는 것이 아니라 그와 연결된 모든 전체 경로인 sub-graph를 더 연결해 주어야 함
                  nodes_to_explore_A = [out_edge[1]]
                  # print(nodes_to_explore_A) #test
                  # import time #test
                  # time.sleep(10) #test
                  explored_nodes_A = set()
                  explored_nodes_A.add(node)

                  # BFS(너비 우선 탐색) 방식으로 연결된 서브그래프 모두 탐색
                  while nodes_to_explore_A:
                      current_node = nodes_to_explore_A.pop(0)
                      explored_nodes_A.add(current_node)

                      # 현재 노드에서 나가는 모든 엣지를 따라 서브그래프 확장
                      for successor in A.successors(current_node):
                        # 동일한 edge가 그래프에 안들어가있으면 edge는 추가
                        # 동일한 successor가 explored_nodes_A에 있으면 needs_to_explore_A에 추가X
                          if successor not in explored_nodes_A:
                              subgraph.add_edge(current_node, successor, relation=A.edges[(current_node, successor)]['relation'])
                              nodes_to_explore_A.append(successor)

                      # 현재 노드로 들어오는 엣지도 추가
                      for predecessor in A.predecessors(current_node):
                          if predecessor not in explored_nodes_A:
                              subgraph.add_edge(predecessor, current_node, relation=A.edges[(predecessor, current_node)]['relation'])
                              nodes_to_explore_A.append(predecessor)


                  # 해당 노드로 들어오는 edge들에 연결된 모든 서브그래프 추가
                  for in_edge in A.in_edges(node):
                      subgraph.add_edge(in_edge[0], node, relation=A.edges[in_edge]['relation'])
                  # in_edge[0]에서부터 연결된 모든 상위 노드와 엣지들도 추가 (들어오는 방향)
                      nodes_to_explore = [in_edge[0]]
                      explored_nodes = set()
                      explored_nodes.add(node)

                      # BFS 방식으로 들어오는 방향으로 연결된 서브그래프 모두 탐색
                      while nodes_to_explore:
                          current_node = nodes_to_explore.pop(0)
                          explored_nodes.add(current_node)

                          # 현재 노드로 들어오는 모든 엣지를 따라 서브그래프 확장
                          for predecessor in A.predecessors(current_node):
                              if predecessor not in explored_nodes:
                                  subgraph.add_edge(predecessor, current_node, relation=A.edges[(predecessor, current_node)]['relation'])
                                  nodes_to_explore.append(predecessor)

                          # 현재 노드에서 나가는 엣지도 추가 (나가는 방향)
                          for successor in A.successors(current_node):
                              if successor not in explored_nodes:
                                  subgraph.add_edge(current_node, successor, relation=A.edges[(current_node, successor)]['relation'])
                                  nodes_to_explore.append(successor)


                  # 새로 생성된 서브그래프를 분해할 목록에 추가
                  will_decom_graph.append(subgraph)

              split_by_outgoing = True
              break

            # 들어오는 엣지가 2개 이상이고, 동일한 relation이 있는 경우 분해
            elif len(in_edges) > 1:
                # print("test") #test
                relation_groups = {}  # relation별로 그룹화
                for in_edge in in_edges:
                    relation = G.edges[in_edge]['relation']
                    if relation not in relation_groups:
                        relation_groups[relation] = []
                    relation_groups[relation].append(in_edge)

                # 동일한 relation을 가진 엣지들로 서브그래프 생성
                for relation, edges in relation_groups.items():
                  # print(edges) #test
                  if len(edges) < 2:
                    continue

                  else:
                    for edge in edges:
                      subgraph = nx.DiGraph()
                      # 동일한 relation을 가진 들어오는 edge 추가
                      subgraph.add_edge(edge[0], node, relation = relation)

                      # edges 배열에 포함되어 있지 않은 edge들과 그 sub-graph들도 node와 연결
                      # subgraph에 포함되지 않는 모든 들어오는 edge에 연결된 subgraph 연결
                      # 해당 relation을 가지지 않은, 들어오는 다른 edge들 모두 추가해야 함.
                      nodes_to_explore_C = [node]
                      explored_nodes_C = set()
                      explored_nodes_C.add(node)
                      explored_edges_C = set(edges)
                      explored_edges_C.discard(edge)

                      while nodes_to_explore_C:
                        current_node = nodes_to_explore_C.pop(0)
                        explored_nodes_C.add(current_node)

                        # 현재 노드로 들어오는 다른 edge를 모두 추가
                        for in_edge_other in A.in_edges(current_node):
                          if in_edge_other not in explored_edges_C:
                            subgraph.add_edge(in_edge_other[0], current_node, relation=A.edges[in_edge_other]['relation'])
                            explored_edges_C.add(in_edge_other)
                            if in_edge_other[0] not in explored_nodes_C:
                                nodes_to_explore_C.append(in_edge_other[0])

                        # 현재 노드에서 나가는 edge들도 모두 추가
                        for out_edge_other in A.out_edges(current_node):
                          if out_edge_other not in explored_edges_C:
                            subgraph.add_edge(current_node, out_edge_other[1], relation=A.edges[out_edge_other]['relation'])
                            explored_edges_C.add(out_edge_other)
                            if out_edge_other[1] not in explored_nodes_C:
                                nodes_to_explore_C.append(out_edge_other[1])

                      # 모든 나가는 edge에 연결된 subgraph 연결
                      for out_edge in A.out_edges(node):
                        subgraph.add_edge(node, out_edge[1], relation=A.edges[out_edge]['relation'])

                        nodes_to_explore_D = [out_edge[1]]
                        # print(nodes_to_explore_D) #test
                        explored_nodes_D = set()
                        explored_nodes_D.add(node)

                        # BFS(너비 우선 탐색) 방식으로 연결된 서브그래프 모두 탐색
                        while nodes_to_explore_D:
                            current_node = nodes_to_explore_D.pop(0)
                            explored_nodes_D.add(current_node)

                            # 현재 노드에서 나가는 모든 엣지를 따라 서브그래프 확장
                            for successor in A.successors(current_node):
                              # 동일한 edge가 그래프에 안들어가있으면 edge는 추가*****
                              # 동일한 successor가 explored_nodes_A에 있으면 needs_to_explore_A에 추가X
                                if successor not in explored_nodes_D:
                                    subgraph.add_edge(current_node, successor, relation=A.edges[(current_node, successor)]['relation'])
                                    nodes_to_explore_D.append(successor)

                            # 현재 노드로 들어오는 엣지도 추가
                            for predecessor in A.predecessors(current_node):
                                if predecessor not in explored_nodes_D:
                                    subgraph.add_edge(predecessor, current_node, relation=A.edges[(predecessor, current_node)]['relation'])
                                    nodes_to_explore_D.append(predecessor)

                      will_decom_graph.append(subgraph)


                    split_by_incoming = True
                    break

        # 나가는 엣지가 2개 이상인 노드가 없고, 들어오는 엣지에 의한 분해도 없으면 전체 그래프 반환
        if not split_by_incoming and not split_by_outgoing:
          subgraphs.append(A)

      return subgraphs

  # 생성된 서브그래프 가져오기
  subgraphs = create_subgraphs(G)

  # subgraph에서 tripleset 추출
  Final_triplesets = []
  def extract_triplesets(graph):
      triplesets = []
      for u, v, data in graph.edges(data=True):
          subject = u
          predicate = data.get('relation', 'relation')
          obj = v
          triplesets.append((subject, predicate, obj))
      return triplesets

  for SubGraph in subgraphs:
      if SubGraph.number_of_nodes() > 1:
        Final_triplesets.append(extract_triplesets(SubGraph))

  temp_decom_question = []
  for tr in Final_triplesets:
      print(tr)
      triples_text = ', '.join([f"('{triple[0]}', '{triple[1]}', '{triple[2]}')" for triple in tr])
      print(triples_text)

      messages = [
          {"role": "system", "content": graph_to_text},
          {"role": "user", "content": triples_text}
      ]

      output = process_data(messages)
      print(output)
      decomposed_queries.append(output)

  return decomposed_queries
