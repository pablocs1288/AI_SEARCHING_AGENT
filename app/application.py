import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(str(current_dir))

import logging

from flask import Flask, request, jsonify

import src.utils.hugging_face_utils as  hugging_face_utils
from src.agent.conditional_logic_agent import LogicAgent
#from src.agent.rewoo_agent import ReWOOAgent




# Flask app
app = Flask(__name__)
with app.app_context():
    app.config["BASE_DIR"] = os.path.join(os.path.realpath(__file__))


# logger configuration
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# hugging face login for downloading llms
logger.info('logging in into huggingface...')
hugging_face_utils.logging_in() # if it is the first time the container is running,  to download the models executed locally a login to huggingface must be done. Works on secrets (test if this needs to be done once at the beginning or at ewach agent calling)
logger.info('login succeed!')


@app.route("/logic_search", methods=("GET", 'POST'))
def logic_search():
    
    try:
        board_members = ''
        if request.method == 'POST':
            query_company_name = request.args.get('company_name')
            logger.info(f"[POST] Company name: {query_company_name}")
            if query_company_name:
                agent = LogicAgent(logger)
                board_members = agent.run_agent(query_company_name)
            else:
                board_members = ''
      
            return jsonify({'board_members': board_members})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    