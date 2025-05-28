conda init
source ~/.bashrc
export REQUESTS_CA_BUNDLE='/opt/conda/lib/python3.11/site-packages/certifi/cacert.pem'

nvidia-smi


ldconfig /.singularity.d/libs
#python /project/6080355/lingjzhu/llm/serve.py
#python /project/6080355/lingjzhu/llm/process_characters_vllm.py
#python /project/6080355/lingjzhu/llm/process_characters_vllm.py
#python /project/6080355/lingjzhu/llm/process_characters_formatted_vllm.py
#python /project/6080355/lingjzhu/llm/process_character_names_formatted_vllm.py
python /project/6080355/lingjzhu/llm/get_character_embeddings.py
