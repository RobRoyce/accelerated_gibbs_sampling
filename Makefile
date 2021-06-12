CC=gcc -g
CFLAGS=--std=c99 -Wall -Wpedantic -lm
INIT := init
TESTS := cont_pdf_test cont_gof_test int_test 
MODULES := distrs utils gmm gmm_gibbs

.PHONY : clean test

all: $(INIT) $(MODULES) $(TESTS)

init:
	mkdir -p obj bin

distrs: src/distrs.c src/distrs.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

utils: src/utils.c src/utils.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

gmm: src/gmm.c src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

gmm_gibbs: src/gmm_gibbs.c src/gmm.h
	$(CC) -c $(CFLAGS) $< -o obj/$@.o

cont_pdf_test: test/cont_pdf_test.c
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

cont_gof_test: test/cont_gof_test.c
#	$(CC) $(CFLAGS) obj/* $^ -o bin/$@
#	./bin/$@

int_test: test/int_test.c
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=2 obj/* $^ -o bin/$@-1 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=4 obj/* $^ -o bin/$@-2 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=8 obj/* $^ -o bin/$@-3 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=16 obj/* $^ -o bin/$@-4 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=32 obj/* $^ -o bin/$@-5 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=64 obj/* $^ -o bin/$@-6 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=128 obj/* $^ -o bin/$@-7 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=2 obj/* $^ -o bin/$@-8 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=4 obj/* $^ -o bin/$@-9 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=8 obj/* $^ -o bin/$@-10 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=16 obj/* $^ -o bin/$@-11 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=32 obj/* $^ -o bin/$@-12 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=64 obj/* $^ -o bin/$@-13 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=128 obj/* $^ -o bin/$@-14 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=2 obj/* $^ -o bin/$@-15 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=4 obj/* $^ -o bin/$@-16 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=8 obj/* $^ -o bin/$@-17 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=16 obj/* $^ -o bin/$@-18 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=32 obj/* $^ -o bin/$@-19 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=64 obj/* $^ -o bin/$@-20 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=128 obj/* $^ -o bin/$@-21 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=2 obj/* $^ -o bin/$@-22 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=4 obj/* $^ -o bin/$@-23 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=8 obj/* $^ -o bin/$@-24 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=16 obj/* $^ -o bin/$@-25 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=32 obj/* $^ -o bin/$@-26 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=64 obj/* $^ -o bin/$@-27 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=128 obj/* $^ -o bin/$@-28 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=2 obj/* $^ -o bin/$@-29 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=4 obj/* $^ -o bin/$@-30 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=8 obj/* $^ -o bin/$@-31 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=16 obj/* $^ -o bin/$@-32 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=32 obj/* $^ -o bin/$@-33 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=64 obj/* $^ -o bin/$@-34 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=128 obj/* $^ -o bin/$@-35 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=2 obj/* $^ -o bin/$@-36 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=4 obj/* $^ -o bin/$@-37 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=8 obj/* $^ -o bin/$@-38 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=16 obj/* $^ -o bin/$@-39 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=32 obj/* $^ -o bin/$@-40 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=64 obj/* $^ -o bin/$@-41 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=128 obj/* $^ -o bin/$@-42 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=2 obj/* $^ -o bin/$@-43 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=4 obj/* $^ -o bin/$@-44 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=8 obj/* $^ -o bin/$@-45 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=16 obj/* $^ -o bin/$@-46 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=32 obj/* $^ -o bin/$@-47 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=64 obj/* $^ -o bin/$@-48 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=128 obj/* $^ -o bin/$@-49 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=2 obj/* $^ -o bin/$@-50 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=4 obj/* $^ -o bin/$@-51 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=8 obj/* $^ -o bin/$@-52 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=16 obj/* $^ -o bin/$@-53 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=32 obj/* $^ -o bin/$@-54 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=64 obj/* $^ -o bin/$@-55 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=128 obj/* $^ -o bin/$@-56 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=2 obj/* $^ -o bin/$@-57 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=4 obj/* $^ -o bin/$@-58 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=8 obj/* $^ -o bin/$@-59 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=16 obj/* $^ -o bin/$@-60 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=32 obj/* $^ -o bin/$@-61 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=64 obj/* $^ -o bin/$@-62 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=128 obj/* $^ -o bin/$@-63 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=2 obj/* $^ -o bin/$@-64 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=4 obj/* $^ -o bin/$@-65 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=8 obj/* $^ -o bin/$@-66 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=16 obj/* $^ -o bin/$@-67 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=32 obj/* $^ -o bin/$@-68 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=64 obj/* $^ -o bin/$@-69 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=128 obj/* $^ -o bin/$@-70 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=2 obj/* $^ -o bin/$@-71 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=4 obj/* $^ -o bin/$@-72 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=8 obj/* $^ -o bin/$@-73 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=16 obj/* $^ -o bin/$@-74 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=32 obj/* $^ -o bin/$@-75 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=64 obj/* $^ -o bin/$@-76 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=128 obj/* $^ -o bin/$@-77 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=2 obj/* $^ -o bin/$@-78 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=4 obj/* $^ -o bin/$@-79 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=8 obj/* $^ -o bin/$@-80 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=16 obj/* $^ -o bin/$@-81 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=32 obj/* $^ -o bin/$@-82 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=64 obj/* $^ -o bin/$@-83 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=128 obj/* $^ -o bin/$@-84 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=2 obj/* $^ -o bin/$@-85 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=4 obj/* $^ -o bin/$@-86 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=8 obj/* $^ -o bin/$@-87 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=16 obj/* $^ -o bin/$@-88 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=32 obj/* $^ -o bin/$@-89 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=64 obj/* $^ -o bin/$@-90 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=128 obj/* $^ -o bin/$@-91 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=2 obj/* $^ -o bin/$@-92 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=4 obj/* $^ -o bin/$@-93 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=8 obj/* $^ -o bin/$@-94 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=16 obj/* $^ -o bin/$@-95 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=32 obj/* $^ -o bin/$@-96 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=64 obj/* $^ -o bin/$@-97 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=128 obj/* $^ -o bin/$@-98 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=2 obj/* $^ -o bin/$@-99 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=4 obj/* $^ -o bin/$@-100 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=8 obj/* $^ -o bin/$@-101 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=16 obj/* $^ -o bin/$@-102 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=32 obj/* $^ -o bin/$@-103 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=64 obj/* $^ -o bin/$@-104 -lm
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=128 obj/* $^ -o bin/$@-105 -lm

clean:
	rm -rf bin
	rm -rf obj

test:
	bash test/run_tests.sh