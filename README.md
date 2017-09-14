Steps:

 1. python test.py
 1. MODEL_DIR=test/`ls test/|tail -n1`
 1. saved_model_cli show --dir ${MODEL_DIR} --all
 1. gcloud ml-engine local predict --model-dir=${MODEL_DIR} --json-instances=instances.json
