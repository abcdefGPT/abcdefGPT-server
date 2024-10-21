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
Extract all relationships that have a directional relationship between entities.
Here, relationship refers to a variety of relationships in which one entity describes, is the same, or includes another, describe another etc.
Format: Entity1 | Entity2\nEntity1 | Entity2 ...
Entity1 segment/narrow down Entity2 or Entity1 equals to Entity2
Entity2 is explains/describes/influence/perform Entity1.
Don't extract the comparison between Entity1 and Entity2
Please extract all possible relationships. Every meaning in the Question should be included, and not left out.
You have to extract all the implicit and explicit relationships, include multiple relationships.

e.g.)
Question: How has climate change, driven by human activity, impacted biodiversity and endangered species?
List of Entities: How, climate change, human activity, biodiversity, endangered species

->
human activity | climate change
climate change | biodiversity
climate change | endangered species

---
e.g.)
Question: Which country is richer, Germany or France?
List of Entities: Which country, Germany, France

->
Which country | Germany
Which Country | France

---
e.g.)
Question: What is the country where the renowned physicist Albert Einstein conducted his groundbreaking research in theoretical physics and developed the theory of general relativity during the early 20th century despite facing significant political and social challenges?
List of Entities: What, country, physicist Albert Einstein, groundbreaking research, theoretical physics, theory, general relativity, early 20th century, significant political challenges, social challenges

->
country | physicist Albert Einstein
physicist Albert Einstein | groundbreaking research
groundbreaking research | theoretical physics
physicist Albert Einstein | theory
theory | general relativity
general relativity | early 20th century
early 20th century | significant political challenges
early 20th century | social challenges
"""

"""
e.g.)
Question: The university where Alan Turing studied during World War II is located in which country?
List of Entities: The university, Alan Turing, World War II, country

->
The university | Alan Turing
Alan Turing | World War II
World War II | country
---
e.g.)
Question: What is the name of the mascot of the university whose main campus is in Columbus, Ohio, and whose branch campuses are scattered throughout the state?
List of Entities: What, name, mascot, university, main campus, Columbus, Ohio, branch campuses, state

->
name | mascot
mascot | university
university | main campus
main campus | Columbus
Columbus | Ohio
university | branch campuses
branch campuses | state
"""

relation_extraction_prompt = """
You are an expert in identifying relationships between input entity pairs in a sentence.
When extract relation, follow these guidelines:
1. Given a sentence and pairs of entities, extract the relation between two entities in each pair of entities based solely on the information provided in the sentence.
2. Extract relationships between entities from the given sentence using the original phrasing as much as possible. But, When encountering relations that imply equivalence (e.g., "equal," "is," "same as"), always express the relation as "equals to.". If extracted information reflects the direction bidirectional or equivalence relationship, in which case use "equals to."
3. For each entity pair, provide the relationship between Entity1 and Entity2. You must always provide a relation for each pair, no matter how minimal or implicit the connection might be. It is Important!!
4. Don't create any additional entity pairs.
5. Keep the order and form of the input entities exactly as they are given. It is important!!
6. There must always be a relation between the entities, even if it is implicit.

Input format:
Sentence: Sentence
Entity pairs:
Entity1 | Entity2
Entity1 | Entity2 ...

Output format:
Entity1 | relationship | Entity2
Entity1 | relationship | Entity2 ...
The output must strictly follow this format.

e.g.)
Sentence: Which writer, known for his work on 'Jurassic Park', co-wrote a film starring Jeff Goldblum and Laura Dern?
Entity pairs:
writer | 'Jurassic Park'
writer | film
film | Jeff Goldblum
film | Laura Dern

->
writer | known for his work on | 'Jurassic Park'
writer | co-wrote | film
film | starring | Jeff Goldblum
film | starring | Laura Dern
"""

NER_prompt = """
You are a capable entity extractor.
You need to extract all Entities from the given sentence.
When extract entity, follow these guidelines:
1. Entities in all noun forms must be extracted.
2. Extracts all entities with explicitly stated meanings in sentences. Extract entities as specifically as possible without duplicating.
3. All Entities should be individually meaningful, You shouldn't extract meaningless Entities such as Be verbs
4. if a relationship is not explicitly stated, connect and extract related entities. if there is no relationship between entities, list them separately.
   - Entities should be connected based on their semantic relationship or if they belong to the same category (e.g., nationality -> American).
   - Avoid connecting entities where the relationship is unclear or ambiguous.
5. interrogative word must should be treated as an Entity.
All Entities should be extracted in the form of Entities, Entities, Entities.
Over-extracting is better than missing out.
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
Question: Is Kelly coming to the party tonight?

->
Kelly, party, tonight
"""

graph_to_text = """You are an experienced text generator.
Please create a complete sentence using all the triples given.
The text will be provided in the format (entity, relation, entity).
Do not add anything else.

e.g.)
Triplesets: [('Sam', 'received', 'award'), ('award', 'from', 'the Academy awards')]

->
Sam received an award from the Academy awards
"""