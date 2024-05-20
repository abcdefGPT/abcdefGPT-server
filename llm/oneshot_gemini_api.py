import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

system_prompt = "You need to get multiple single questions by performing decomposition to make it easier to search for multiple documents from complex questions. " \
                "The conditions are as follows. Please make sure that all conditions are met." \
                "1. The complex question (Q) should be decomposed into multiple single questions (SQs)." \
                "2. Decomposition should be consistent." \
                "3. The decomposed SQs should be generated based on the common main entity in Q." \
                "4. All SQs should closely relate to the original Q for effective document retrieval." \
                "5. Avoid generating SQs that are too general or unrelated to the specific context of Q."

user_prompt = "Q: Did the article from The Verge about Hasbro's Jenga: Super Mario Edition and the article from Engadget about the '9th generation iPad' both report a discount on their respective products during the Black Friday sales on Amazon?"

print(user_prompt)

response = model.generate_content(system_prompt + " " + user_prompt)

print(response.text)