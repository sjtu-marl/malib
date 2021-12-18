#.PHONY: test-training-agent-interface
#test-training-agent-interface:
#	PYTHONHASHSEED=24 pytest -v
#
#.PHONY: test-learning
#test-learning:
#
#.PHONY: profiling
#profiling:

.PHONY: clean
clean:
	rm -rf ./logs/*

.PHONY: format
format:
	# pip install black==20.8b1
	black .

.PHONY: docs
docs:
	(cd docs; make clean html)

.PHONY: rm-pycache
rm-pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: test
test:
	pytest --cov-config=.coveragerc --cov=malib --cov-report html --doctest-modules tests
	rm -f .coverage.*

.PHONY: test-dataset
test-dataset:
	pytest -v --doctest-modules tests/dataset

.PHONY: test-parameter-server
test-parameter-server:
	pytest -v --doctest-modules tests/paramter_server

.PHONY: test-coordinator
test-coordinator:
	pytest -v --doctest-modules tests/coordinator

.PHONY: test-backend
test-backend: test-dataset test-parameter-server test-coordinator

.PHONY: test-algorithm
test-algorithm:
	pytest -v --doctest-modules tests/algorithm

.PHONY: test-rollout
test-rollout:
	pytest -v --doctest-modules tests/rollout

.PHONY: test-agent
test-agent:
	pytest --doctest-modules tests/agent

.PHONY: test-env-api
test-env-api:
	pytest -v --doctest-modules test-env-api

