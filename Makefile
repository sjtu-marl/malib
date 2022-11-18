#.PHONY: test-training-agent-interface
#test-training-agent-interface:
#	PYTHONHASHSEED=24 pytest -v
#
#.PHONY: test-learning
#test-learning:
#
#.PHONY: profiling
#profiling:
# run visualization for cov_html: ruby -run -ehttpd . -p8000
.PHONY: clean
clean:
	rm -rf ./logs/*

.PHONY: format
format:
	# pip install black==20.8b1
	black .

.PHONY: docs-compile
docs-compile:
	(cd docs; make clean html)

.PHONY: docs-view
docs-view:
	python -m http.server 8000 -d docs/build/html/

.PHONY: rm-pycache
rm-pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: test
test:
	pytest --cov-config=.coveragerc --cov=malib --cov-report term --cov-report html --doctest-modules tests
	rm -f .coverage.*

.PYTHON: test-rollout
test-rollout:
	pytest --cov-config=.coveragerc --cov=malib/rollout tests/rollout
	rm -f .coverage.*

.PYTHON: test-rl
test-rl:
	pytest --cov-config=.coveragerc --cov=malib/rl tests/rl
	rm -f .coverage.*

.PYTHON: test-backend
test-backend:
	pytest --cov-config=.coveragerc --cov=malib/backend tests/backend
	rm -f .coverage.*

.PYTHON: test-agents
test-agents:
	pytest --cov-config=.coveragerc --cov=malib/agent tests/agents
	rm -f .coverage.*

.PYTHON: coverage-view
coverage-view:
	python -m http.server 8000 -d cov_html

.PHONY: test-verbose
test-verbose:
	pytest --cov-config=.coveragerc --cov=malib --cov-report html --doctest-modules tests -v -s
	rm -f .coverage.*
