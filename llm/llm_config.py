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

# file_path = os.getcwd() + "llm/vector_store/hotpotqa_test_compressed_vectorstore_ver2.index"
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


# Define prompt template
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise, limited to one or two words
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Setup RAG pipeline
rag_chain = (
    {"context": RunnablePassthrough(),  "question": RunnablePassthrough()}
    | prompt
    | answer_llm
    | StrOutputParser()
)

RE_prompt = """
You are a competent relationship extractor.
The goal is to extract the meaning in the query.
Please extract relationships between entities. However, you should extract only the knowledge that exists in the query, not the relationship between entities using existing knowledge.
Please extract the relation with the words that exist in the query as much as possible.
All relationships should be extracted in the form of Entity|Relationship|Entity.
Only relation must be in the middle
Necessarily Do not use any entity other than that given. Don't change the entity as you please.
The given entity should be extracted separately for each entity.
Interrogative should be included as entity.
Don't print anything other than what you asked

e.g.)
Question: What measures might the international community take if X (formerly Twitter) fails to comply with the European Union's Code?
List of Entities: What, measures, international community, X (formerly Twitter), European Union's Code

->
What|to|measures
international community|take|measures
measures|assume|X (formerly Twitter)
X (formerly Twitter)|fails to comply with|European Union's Code

e.g. )
Question: Is the legal battle between Google and Epic Games unprecedented?
List of Entities: legal battle, Google and Epic Games

->
legal battle|between|Google and Epic Games

e.g. )
Question: Who was the Super Bowl MVP in 1979 and 1980.
List of Entities: Who, Super Bowl MVP, 1979 and 1980

->
Who|was|Super Bowl MVP
Super Bowl MVP|in|1979 and 1980
"""

NER_prompt = """
You are a capable entity extractor.
You need to extract all Entities from the given sentence (query, question).
You need to extract all the entities that have meaning in the sentence.
All Entities should be extracted in the form of Entities, Entities, Entities.
When extracting entity, extract as much as possible.
interrogative word must should be treated as an entity Where was it located.
Don't print anything other than what you asked

e.g. )
Question: What measures might the international community take if X (formerly Twitter) fails to comply with the European Union's Code?

->
What, measures, international community, X (formerly Twitter), European Union's Code

e.g. )
Question: Who was the Super Bowl MVP in 1979 and 1980.

->
Who, Super Bowl MVP, 1979 and 1980

e.g. )
Question: 2014 S/S is the debut album of a South Korean boy group that was formed by who?

->
2014 S/S, debut album, South Korean boy group, who
"""

#core
system_message = """
You are an experienced Named Entity Recognizer.
Your task is to extract entities from the given text and classify them as either 'core' or 'etc'.


The reason for extracting 'core' and 'etc' entities is to facilitate query decomposition. By recognizing 'core' and 'etc' entities, the query can be divided into multiple sub-queries for more precise information retrieval. For example, the query "Who was the Super Bowl MVP in 1979 and 1980?" can be decomposed into "Who was the Super Bowl MVP in 1979?" and "Who was the Super Bowl MVP in 1980?" based on the identified entities: Super Bowl MVP (core), 1979 (etc), and 1980 (etc).

- 'core' entities are the main subjects or topics that are consistent across variations of the query.The query will be divided based on this entity.
- 'etc' entities represent others except for 'core'

We're going to divide the query based on 'core',
so it's a condition that one entity has to have a relationship with two or more entities to become 'core'.

Furthermore, the ultimate goal is to extract entities from the query, use them to create nodes, and build a knowledge graph. This knowledge graph will enable more efficient and accurate information retrieval and analysis.
If it is judged that there is no 'core', you can judge all entities as 'etc'.


I'll provide the query, the entities, and the relationship between them.
Write the entities in the following format: EntityName (tag).



Example:
Query: Who was the Super Bowl MVP in 1979 and 1980?\nEntity: Super Bowl MVP, 1979, 1980\nSuper Bowl MVP|was|1979\nSuper Bowl MVP|was|1980
You:
Super Bowl MVP (core), 1979 (etc), 1980 (etc)

Example:
Query: Are Chris Marker and Yvonne Rainer American?\nEntity: Chris Marker, Yvonne Rainer, American\nChris Marker|is|American\nYvonne Rainer|is|American
You:
Chris Marker (etc), Yvonne Rainer (etc), American (core)

Example:
Query: How has Elon Musk's acquisition of X (formerly Twitter) impacted the stock prices and market valuation of related companies such as Tesla and SpaceX?\nEntity: Elon Musk's acquisition, X (formerly Twitter), stock prices, market valuation, Tesla , SpaceX\nElon Musk's acquisition|impacts|stock prices\nElon Musk's acquisition|impacts|market valuation\nElon Musk's acquisition|impacts|Tesla\nElon Musk's acquisition|impacts|SpaceX
You:
Elon Musk's acquisition (etc), X (formerly Twitter) (etc), stock prices (etc), market valuation of related companies (core), Tesla (etc), SpaceX (etc)

Example:
Query: Who directed The Godfather and Apocalypse Now?\nEntity: Who, The Godfather, Apocalypse Now\nWho|directed|The Godfather\nWho|directed|Apocalypse Now
You:
Who (core), The Godfather (etc), Apocalypse Now (etc)

Treat possessive forms (e.g., Mary's) as part of the same entity.
Do not add anything else.

Keep in mind that you can't be a 'core' if you don't get that entity more than once in the relationship I give you
"""

graph_to_text = """You are an experienced text generator.
Please create a complete sentence using all the triples given.
The text will be provided in the format (entity, relation, entity).
Do not add anything else.
Example Text: [('Sam', 'received', 'award'), ('award', 'from', 'the Academy awards')]
Answer: Sam received an award from the Academy awards
"""

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
    messages=messages
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
    messages=messages
  )

  return response.choices[0].message.content

# core 뽑기
def get_core(words, relations, query):
  all = "Query: " + query + "\nEntity: " + words + "\n" + relations
  messages = [{
      "role": "system",
      "content": system_message
  }, {
      "role": "user",
      "content": all
  }]
  response = openai.chat.completions.create(
    model=model,
    messages=messages
  )

  return response.choices[0].message.content

def convert_core(q1_before):
    # 각 항목을 분리하여 리스트로 변환
    components = [comp.strip() for comp in q1_before.split(',')]

    # 엔티티와 속성 파싱
    entities = []
    for comp in components:
        if '(' in comp and ')' in comp:
            entity, attribute = comp.split('(')
            attribute = attribute.strip(')').strip()
        else:
            entity = comp.strip()
            attribute = 'core'  # 기본 속성으로 설정
        entities.append({'entity': entity.strip(), 'attribute': attribute})

    return entities


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

def query_decomposition(rener_json, cores_json):
  decomposed_queries = []

  G = nx.DiGraph()

  # G에 entity를 node로 추가
  for nodes in rener_json['entities']:
    G.add_node(nodes)

  for nodes_with_attribute in cores_json:
    entity = nodes_with_attribute['entity']
    attribute = nodes_with_attribute['attribute']

    # G에 노드로 존재하는 경우에만 아래 코드 수행, 단 추가가 아니라 type과 color만 변경
    if G.has_node(entity):
        if attribute == "core":
            G.add_node(entity, type=attribute, color="#FF0081")  # 핑크색
        elif attribute == "etc":
            G.add_node(entity, type=attribute, color="#5858FA")  # 파란색
    else:
        G.add_node(entity, type="etc", color="#005DEA")  # 그 외 파란색

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
                # print(relation_groups) #test

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

# async def create_vector_store(app):
#     loader = DirectoryLoader('rule/public_institution_rule_english', glob="*.json", show_progress=True,
#                              loader_cls=TextLoader)
#     docs = loader.load()
#     vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings(api_key=openai.api_key))
#     app.state.vectorstore = vectorstore