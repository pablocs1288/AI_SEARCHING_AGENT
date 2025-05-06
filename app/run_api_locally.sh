export FLASK_ENV="local"
export FLASK_DEBUG=0
export ENV_FOR_DYNACONF="local"
export CHROMA_DB_DIR="./chroma_db"
export LOCAL_NER_MODEL="en_core_web_sm"
export LOCAL_EMBEDDING_MODEL="NovaSearch/stella_en_1.5B_v5"
export LOCAL_LLM_AGENT="microsoft/Phi-3.5-mini-instruct"
export SIMILARITY_THRESHOLD=200


export PYTHONPATH="./"
export FLASK_APP=application.py

flask run --host 0.0.0.0 --port 5000