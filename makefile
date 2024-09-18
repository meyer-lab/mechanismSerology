.PHONY: clean test pyright

flist = $(wildcard maserol/figures/figure*.py)

all: $(patsubst maserol/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: .venv maserol/figures/figure%.py
	mkdir -p output
	rye run fbuild $*

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=maserol --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright maserol

clean:
	rm -rf coverage.xml