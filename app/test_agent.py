import os
import logging


import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(str(current_dir))


from src.agent.conditional_logic_agent import LogicAgent
from src.agent.rewoo_agent import ReWOOAgent


# set env vars
os.environ["CHROMA_DB_DIR"] = './chroma_db'
os.environ["LOCAL_NER_MODEL"] = 'en_core_web_sm'
os.environ["LOCAL_EMBEDDING_MODEL"] = 'NovaSearch/stella_en_1.5B_v5'
os.environ["LOCAL_LLM_AGENT"] = 'microsoft/Phi-3.5-mini-instruct'
os.environ["SIMILARITY_THRESHOLD"] = '200'

# logger configuration
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,  
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger(__name__)

#agent = LogicAgent(logger)
agent = ReWOOAgent(logger)

# Example interaction
if __name__ == "__main__":
    while True:
        company_name = input("Enter company name (or 'exit'): ")
        if company_name == 'exit':
            break

        #members = agent.run_agent(company_name)
        members = agent.run_agent(company_name, verbose = True)
        print(f"Board Members of {company_name}:\n", members)