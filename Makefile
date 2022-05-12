install:
	pip install --upgrade pip &&\
		pip install -r unit-requirements.txt &&\
        pip install -e .
        
lint:
	python -m pylint --fail-under=-200.5 --rcfile .pylintrc cicd_databricks_github/ tests/ -r n --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" > pylint_report.txt      #pylint --disable=R,C model.py 

format:
	black *.py

test:
	python -m pytest -vv --disable-warnings tests/ --junitxml=junit/test-results.xml --cov=. --cov-config=.coveragerc --cov-report xml:coverage.xml --cov-report term #--cov-report html:cov_html #--doctest-modules #--cov=hello test_hello.py
	
validate:
	dbx deploy --jobs=validation --deployment-file=./conf/deployment-validation.json
	dbx launch --job=validation --trace

all: install lint test