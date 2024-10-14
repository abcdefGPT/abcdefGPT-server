# Define prompt template
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise, limited to one or two words
Question: {question}
Context: {context}
Answer:
"""

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