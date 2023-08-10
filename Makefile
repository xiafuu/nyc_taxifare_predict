.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

run_preprocess:
	python -c 'from taxifare.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from taxifare.interface.main import train; train()'

run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

run_api:
	uvicorn taxifare.api.fast:app --reload

##################### TESTS #####################
default:
	cat tests/cloud_training/test_output.txt

test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project

test_gcp_project:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_project_id

test_gcp_bucket:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_bucket_name

test_mlflow_config:
	@pytest \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_model_target_is_mlflow \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_experiment_is_not_null \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_model_name_is_not_null

test_prefect_config:
	@pytest \
	tests/lifecycle/test_prefect.py::TestPrefect::test_prefect_flow_name_is_not_null \
	tests/lifecycle/test_prefect.py::TestPrefect::test_prefect_log_level_is_warning

##################### TESTS TRAIN AT SCALE#####################

test_kitt_train_at_scale:
	@echo "\n 🧪 computing and saving your progress at 'tests/train_at_scale/test_output.txt'...(this can take a while)"
	@pytest tests/train_at_scale -c "./tests/pytest_kitt.ini" 2>&1 > tests/train_at_scale/test_output.txt || true
	@echo "\n 🙏 Please: \n git add tests \n git commit -m 'checkpoint' \n ggpush"

test_preprocess_and_train:
	@pytest \
	tests/train_at_scale/test_clean.py \
	tests/train_at_scale/test_processor_pipeline.py \
	tests/train_at_scale/test_model.py \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_preprocess_and_train \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_pred

test_preprocess_by_chunk:
	@pytest \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_preprocess

test_train_by_chunk:
	@pytest \
	tests/train_at_scale/test_main_local.py::TestMainLocal::test_route_train


##################### TESTS CLOUD TRAINING#####################

test_kitt:
	@echo "\n 🧪 computing and saving your progress at 'tests/cloud_training/test_output.txt'...(this can take a while)"
	@pytest tests/cloud_training -c "./tests/pytest_kitt.ini" 2>&1 > tests/cloud_training/test_output.txt || true
	@echo "\n 🙏 Please: \n git add tests \n git commit -m 'checkpoint' \n ggpush"

test_preprocess:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_preprocess

test_train:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_train

test_evaluate:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_evaluate

test_pred:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_pred

test_main_all: test_preprocess test_train test_evaluate test_pred

test_big_query:
	@pytest \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_big_query_dataset_variable_exists \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_cloud_data_create_dataset \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_cloud_data_create_table

test_vm:
	tests/cloud_training/test_vm.py

################### TEST API ################
test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_is_dict --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_has_key --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_val_is_float --asyncio-mode=strict -W "ignore"

test_api_on_prod:
	pytest \
	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"



################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/mlops
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_1k
	-bq show ${BQ_DATASET}.processed_200k
	-bq show ${BQ_DATASET}.processed_all
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/mlops/data/
	mkdir ~/.lewagon/mlops/data/raw
	mkdir ~/.lewagon/mlops/data/processed
	mkdir ~/.lewagon/mlops/training_outputs
	mkdir ~/.lewagon/mlops/training_outputs/metrics
	mkdir ~/.lewagon/mlops/training_outputs/models
	mkdir ~/.lewagon/mlops/training_outputs/params

reset_local_files_with_csv_solutions: reset_local_files
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_all.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_all.csv

reset_bq_files:
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_1k
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_200k
	-bq rm --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_all
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_1k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_200k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_all

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files
