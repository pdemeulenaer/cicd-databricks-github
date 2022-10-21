install:
	pip install --upgrade pip &&\
		pip install -r unit-requirements.txt &&\
        pip install -e .
        
lint:
	python -m pylint --fail-under=-200.5 --rcfile .pylintrc cicd_databricks_github/ tests/ -r n --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" > pylint_report.txt      #pylint --disable=R,C model.py 

format:
	black cicd_databricks_github/*.py

test:
	python -m pytest -vv --disable-warnings tests/ --junitxml=junit/test-results.xml --cov=. --cov-config=.coveragerc --cov-report xml:coverage.xml --cov-report term #--cov-report html:cov_html #--doctest-modules #--cov=hello test_hello.py


# For executions in command line (for test purpose, on interactive clusters)
train_task: # TODO:
	# dbx deploy --jobs=training --deployment-file=./conf/deployment-training.json
	# dbx launch --job=training --trace
	dbx execute train-workflow --task step-training-task --cluster-name ...	

validate_task: # TODO:
	# dbx deploy --jobs=validation --deployment-file=./conf/deployment-validation.json
	# dbx launch --job=validation --trace
	dbx execute train-workflow --task step-validation-task --cluster-name ...		

inference_task: # TODO:
	# dbx deploy --jobs=cd-infer-job-staging --deployment-file=./conf/deployment.json
	# dbx launch --job=cd-infer-job-staging --trace

# For executions within the CI/CD pipeline
train_workflow:
	# dbx deploy --jobs=training --deployment-file=./conf/deployment-training.json
	# dbx launch --job=training --trace
	dbx deploy train-workflow
	dbx launch train-workflow --trace		

inference: # TODO:
	# dbx deploy --jobs=cd-infer-job-staging --deployment-file=./conf/deployment.json
	# dbx launch --job=cd-infer-job-staging --trace

message:
	echo hello $(foo)

all: install lint test