from itertools import chain

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()

def main():
    print("Hello from agentic-ai!")
    information = """
    Muhammad Yunus (Bengali: মূহাম্মদ ইউনূস; born 28 June 1940) is a Bangladeshi economist and statesman. Yunus pioneered the modern concept of microcredit and microfinance, for which he was awarded the Nobel Peace Prize in 2006. He is the founder of Grameen Bank and the first Bangladeshi to win the Nobel Peace Prize. Following the July Uprising, he was appointed as the 5th chief adviser of Bangladesh, the head of the interim government, serving from 2024 to 2026.[1][2]
    Born in Hathazari, Chittagong, Yunus passed his matriculation and intermediate examinations from Chittagong Collegiate School and Chittagong College, respectively. He completed his BA from University of Dhaka and joined as a lecturer in Chittagong College. He obtained his PhD in economics from Vanderbilt University in the United States.
    After the famine of 1974, Yunus started to work on poverty alleviation in Bangladesh. He began experimenting with microfinance in the late 1970s. In 1983, the Grameen Bank was established. The success of the Grameen microfinance model inspired similar efforts in about 100 developing countries and even in developed countries including the United States.[3] Yunus was awarded the Nobel Peace Prize in 2006 for founding the Grameen Bank and pioneering the concepts of microcredit and microfinance.[4] Yunus has received several other national and international honors, including the United States Presidential Medal of Freedom in 2009 and the Congressional Gold Medal in 2010.
    Yunus has been a vocal advocate for social business, which he defines as a non-loss, non-dividend company designed to address a social problem. He has also been involved in various initiatives to promote social business and entrepreneurship around the world."""
    
    summary_template = """
    given the information {information} about a person, I want to create:
    1. A short summary of the person's life and achievements.
    2. Two interesting facts about the person.
    3. A question that can be asked about the person."""
    
    summary_prompt_template= PromptTemplate(template=summary_template, input_variables=["information"])
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0.0)
    
    chain = summary_prompt_template | llm
    
    response = chain.invoke(input={"information": information})
    
    print(response.content) 

if __name__ == "__main__":
    main()
    