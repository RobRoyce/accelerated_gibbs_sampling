CC=nvcc -g
CFLAGS=-lm -lcudadevrt -lcurand -rdc=true
INIT := init
TESTS := cont_pdf_test cont_gof_test int_test
MODULES := distrs utils gmm gmm_gibbs

.PHONY : clean test

all: $(INIT) $(MODULES) $(TESTS)

init:
	mkdir -p obj bin

distrs: src/distrs.cu src/distrs.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o -dlink

utils: src/utils.cu src/utils.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o -dlink

gmm: src/gmm.cu src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o -dlink

gmm_gibbs: src/gmm_gibbs.cu src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o -dlink

cont_pdf_test: test/cont_pdf_test.cu
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

cont_gof_test: test/cont_gof_test.cu
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

int_test: test/int_test.cu
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=4 -DMSAMPLERS=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=4 -DMSAMPLERS=16 obj/* $^ -o bin/$@-7

clean:
	rm -rf bin
	rm -rf obj

test:
	bash test/run_tests.sh