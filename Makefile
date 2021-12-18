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
