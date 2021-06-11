CC=nvcc
CFLAGS=-lm -lcurand
INIT := init
TESTS := cont_pdf_test cont_gof_test int_test
MODULES := distrs utils gmm gmm_gibbs

.PHONY : clean test

all: $(INIT) $(MODULES) $(TESTS)

init:
	mkdir -p obj bin

distrs: src/distrs.cu src/distrs.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

utils: src/utils.cu src/utils.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

gmm: src/gmm.cu src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

gmm_gibbs: src/gmm_gibbs.cu src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

cont_pdf_test: test/cont_pdf_test.cu
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

cont_gof_test: test/cont_gof_test.cu
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

int_test: test/int_test.cu
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=64 obj/* $^ -o 	bin/$@-1
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=32 obj/* $^ -o 	bin/$@-2
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=16 obj/* $^ -o 	bin/$@-3
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=8 obj/* $^ -o 	bin/$@-4
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=4 obj/* $^ -o 	bin/$@-5
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=2 obj/* $^ -o 	bin/$@-6
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=64 obj/* $^ -o 	bin/$@-7
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=32 obj/* $^ -o 	bin/$@-8
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=16 obj/* $^ -o 	bin/$@-9
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=8 obj/* $^ -o 	bin/$@-10
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=4 obj/* $^ -o 	bin/$@-11
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=2 obj/* $^ -o 	bin/$@-12
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=64 obj/* $^ -o 	bin/$@-13
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=32 obj/* $^ -o 	bin/$@-14
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=16 obj/* $^ -o 	bin/$@-15
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=8 obj/* $^ -o 	bin/$@-16
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=4 obj/* $^ -o 	bin/$@-17
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=2 obj/* $^ -o 	bin/$@-18
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=64 obj/* $^ -o 	bin/$@-19
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=32 obj/* $^ -o 	bin/$@-20
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=16 obj/* $^ -o 	bin/$@-21
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=8 obj/* $^ -o 	bin/$@-22
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=4 obj/* $^ -o 	bin/$@-23
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=2 obj/* $^ -o 	bin/$@-24
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=64 obj/* $^ -o 	bin/$@-25
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=32 obj/* $^ -o 	bin/$@-26
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=16 obj/* $^ -o 	bin/$@-27
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=8 obj/* $^ -o 	bin/$@-28
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=4 obj/* $^ -o 	bin/$@-29
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=2 obj/* $^ -o 	bin/$@-30
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=64 obj/* $^ -o 	bin/$@-31
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=32 obj/* $^ -o 	bin/$@-32
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=16 obj/* $^ -o 	bin/$@-33
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=8 obj/* $^ -o 	bin/$@-34
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=4 obj/* $^ -o 	bin/$@-35
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=2 obj/* $^ -o 	bin/$@-36
	$(CC) $(CFLAGS) -DNSAMPLES=8192 -DKCLASSES=128 obj/* $^ -o 	bin/$@-37
	$(CC) $(CFLAGS) -DNSAMPLES=4086 -DKCLASSES=128 obj/* $^ -o 	bin/$@-38
	$(CC) $(CFLAGS) -DNSAMPLES=2048 -DKCLASSES=128 obj/* $^ -o 	bin/$@-39
	$(CC) $(CFLAGS) -DNSAMPLES=1024 -DKCLASSES=128 obj/* $^ -o 	bin/$@-40
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=128 obj/* $^ -o 	bin/$@-41
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=128 obj/* $^ -o 	bin/$@-42
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=128 obj/* $^ -o 	bin/$@-43
	echo "N, K, Time(usec)" > results.csv
	./bin/$@-1  >> results.csv
	./bin/$@-2  >> results.csv
	./bin/$@-3  >> results.csv
	./bin/$@-4  >> results.csv
	./bin/$@-5  >> results.csv
	./bin/$@-6  >> results.csv
	./bin/$@-7  >> results.csv
	./bin/$@-8  >> results.csv
	./bin/$@-9  >> results.csv
	./bin/$@-10 >> results.csv
	./bin/$@-11 >> results.csv
	./bin/$@-12 >> results.csv
	./bin/$@-13 >> results.csv
	./bin/$@-14 >> results.csv
	./bin/$@-15 >> results.csv
	./bin/$@-16 >> results.csv
	./bin/$@-17 >> results.csv
	./bin/$@-18 >> results.csv
	./bin/$@-19 >> results.csv
	./bin/$@-20 >> results.csv
	./bin/$@-21 >> results.csv
	./bin/$@-22 >> results.csv
	./bin/$@-23 >> results.csv
	./bin/$@-24 >> results.csv
	./bin/$@-25 >> results.csv
	./bin/$@-26 >> results.csv
	./bin/$@-27 >> results.csv
	./bin/$@-28 >> results.csv
	./bin/$@-29 >> results.csv
	./bin/$@-30 >> results.csv
	./bin/$@-31 >> results.csv
	./bin/$@-32 >> results.csv
	./bin/$@-33 >> results.csv
	./bin/$@-34 >> results.csv
	./bin/$@-35 >> results.csv
	./bin/$@-36 >> results.csv
	./bin/$@-37 >> results.csv
	./bin/$@-38 >> results.csv
	./bin/$@-39 >> results.csv
	./bin/$@-40 >> results.csv
	./bin/$@-41 >> results.csv
	./bin/$@-42 >> results.csv
	./bin/$@-43 >> results.csv
clean:
	rm -rf bin
	rm -rf obj

test:
	cd bin && ./cont_gof_test && ./int_test && ./cont_pdf_test
