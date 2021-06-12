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
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=2 obj/* $^ -o bin/$@-1
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=4 obj/* $^ -o bin/$@-2
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=8 obj/* $^ -o bin/$@-3
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=16 obj/* $^ -o bin/$@-4
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=32 obj/* $^ -o bin/$@-5
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=64 obj/* $^ -o bin/$@-6
	$(CC) -c $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=128 obj/* $^ -o bin/$@-7
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=2 obj/* $^ -o bin/$@-8
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=4 obj/* $^ -o bin/$@-9
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=8 obj/* $^ -o bin/$@-10
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=16 obj/* $^ -o bin/$@-11
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=32 obj/* $^ -o bin/$@-12
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=64 obj/* $^ -o bin/$@-13
	$(CC) -c $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=128 obj/* $^ -o bin/$@-14
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=2 obj/* $^ -o bin/$@-15
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=4 obj/* $^ -o bin/$@-16
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=8 obj/* $^ -o bin/$@-17
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=16 obj/* $^ -o bin/$@-18
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=32 obj/* $^ -o bin/$@-19
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=64 obj/* $^ -o bin/$@-20
	$(CC) -c $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=128 obj/* $^ -o bin/$@-21
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=2 obj/* $^ -o bin/$@-22
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=4 obj/* $^ -o bin/$@-23
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=8 obj/* $^ -o bin/$@-24
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=16 obj/* $^ -o bin/$@-25
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=32 obj/* $^ -o bin/$@-26
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=64 obj/* $^ -o bin/$@-27
	$(CC) -c $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=128 obj/* $^ -o bin/$@-28
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=2 obj/* $^ -o bin/$@-29
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=4 obj/* $^ -o bin/$@-30
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=8 obj/* $^ -o bin/$@-31
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=16 obj/* $^ -o bin/$@-32
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=32 obj/* $^ -o bin/$@-33
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=64 obj/* $^ -o bin/$@-34
	$(CC) -c $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=128 obj/* $^ -o bin/$@-35
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=2 obj/* $^ -o bin/$@-36
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=4 obj/* $^ -o bin/$@-37
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=8 obj/* $^ -o bin/$@-38
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=16 obj/* $^ -o bin/$@-39
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=32 obj/* $^ -o bin/$@-40
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=64 obj/* $^ -o bin/$@-41
	$(CC) -c $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=128 obj/* $^ -o bin/$@-42
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=2 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=2 obj/* $^ -o bin/$@-43
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=4 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=4 obj/* $^ -o bin/$@-44
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=8 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=8 obj/* $^ -o bin/$@-45
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=16 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=16 obj/* $^ -o bin/$@-46
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=32 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=32 obj/* $^ -o bin/$@-47
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=64 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=64 obj/* $^ -o bin/$@-48
	$(CC) -c $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=128 src/gmm_gibbs.cu -o obj/gmm_gibbs.o -dlink
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=128 obj/* $^ -o bin/$@-49

clean:
	rm -rf bin
	rm -rf obj

test:
	bash test/run_tests.sh