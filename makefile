
flist = $(wildcard maserol/figures/figure*.py)

all: $(patsubst maserol/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: maserol/figures/figure%.py
	mkdir -p output
	poetry run fbuild $*

test:
	poetry run pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports maserol

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x

testcover:
	poetry run pytest --cov=syserol --cov-report=xml --cov-config=.github/workflows/coveragerc

clean:
	rm -rf output
