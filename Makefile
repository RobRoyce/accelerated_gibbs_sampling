CC=nvcc
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
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=2 obj/* $^ -o bin/$@-1
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=4 obj/* $^ -o bin/$@-2
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=8 obj/* $^ -o bin/$@-3
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=16 obj/* $^ -o bin/$@-4
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=32 obj/* $^ -o bin/$@-5
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=64 obj/* $^ -o bin/$@-6
	$(CC) $(CFLAGS) -DNSAMPLES=4  -DKCLASSES=128 obj/* $^ -o bin/$@-7
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=2 obj/* $^ -o bin/$@-8
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=4 obj/* $^ -o bin/$@-9
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=8 obj/* $^ -o bin/$@-10
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=16 obj/* $^ -o bin/$@-11
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=32 obj/* $^ -o bin/$@-12
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=64 obj/* $^ -o bin/$@-13
	$(CC) $(CFLAGS) -DNSAMPLES=8  -DKCLASSES=128 obj/* $^ -o bin/$@-14
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=2 obj/* $^ -o bin/$@-15
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=4 obj/* $^ -o bin/$@-16
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=8 obj/* $^ -o bin/$@-17
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=16 obj/* $^ -o bin/$@-18
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=32 obj/* $^ -o bin/$@-19
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=64 obj/* $^ -o bin/$@-20
	$(CC) $(CFLAGS) -DNSAMPLES=16  -DKCLASSES=128 obj/* $^ -o bin/$@-21
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=2 obj/* $^ -o bin/$@-22
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=4 obj/* $^ -o bin/$@-23
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=8 obj/* $^ -o bin/$@-24
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=16 obj/* $^ -o bin/$@-25
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=32 obj/* $^ -o bin/$@-26
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=64 obj/* $^ -o bin/$@-27
	$(CC) $(CFLAGS) -DNSAMPLES=32  -DKCLASSES=128 obj/* $^ -o bin/$@-28
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=2 obj/* $^ -o bin/$@-29
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=4 obj/* $^ -o bin/$@-30
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=8 obj/* $^ -o bin/$@-31
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=16 obj/* $^ -o bin/$@-32
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=32 obj/* $^ -o bin/$@-33
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=64 obj/* $^ -o bin/$@-34
	$(CC) $(CFLAGS) -DNSAMPLES=64  -DKCLASSES=128 obj/* $^ -o bin/$@-35
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=2 obj/* $^ -o bin/$@-36
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=4 obj/* $^ -o bin/$@-37
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=8 obj/* $^ -o bin/$@-38
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=16 obj/* $^ -o bin/$@-39
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=32 obj/* $^ -o bin/$@-40
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=64 obj/* $^ -o bin/$@-41
	$(CC) $(CFLAGS) -DNSAMPLES=128  -DKCLASSES=128 obj/* $^ -o bin/$@-42
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=2 obj/* $^ -o bin/$@-43
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=4 obj/* $^ -o bin/$@-44
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=8 obj/* $^ -o bin/$@-45
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=16 obj/* $^ -o bin/$@-46
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=32 obj/* $^ -o bin/$@-47
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=64 obj/* $^ -o bin/$@-48
	$(CC) $(CFLAGS) -DNSAMPLES=256  -DKCLASSES=128 obj/* $^ -o bin/$@-49
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=2 obj/* $^ -o bin/$@-50
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=4 obj/* $^ -o bin/$@-51
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=8 obj/* $^ -o bin/$@-52
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=16 obj/* $^ -o bin/$@-53
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=32 obj/* $^ -o bin/$@-54
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=64 obj/* $^ -o bin/$@-55
	$(CC) $(CFLAGS) -DNSAMPLES=512  -DKCLASSES=128 obj/* $^ -o bin/$@-56
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=2 obj/* $^ -o bin/$@-57
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=4 obj/* $^ -o bin/$@-58
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=8 obj/* $^ -o bin/$@-59
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=16 obj/* $^ -o bin/$@-60
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=32 obj/* $^ -o bin/$@-61
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=64 obj/* $^ -o bin/$@-62
	$(CC) $(CFLAGS) -DNSAMPLES=1024  -DKCLASSES=128 obj/* $^ -o bin/$@-63
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=2 obj/* $^ -o bin/$@-64
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=4 obj/* $^ -o bin/$@-65
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=8 obj/* $^ -o bin/$@-66
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=16 obj/* $^ -o bin/$@-67
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=32 obj/* $^ -o bin/$@-68
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=64 obj/* $^ -o bin/$@-69
	$(CC) $(CFLAGS) -DNSAMPLES=2048  -DKCLASSES=128 obj/* $^ -o bin/$@-70
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=2 obj/* $^ -o bin/$@-71
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=4 obj/* $^ -o bin/$@-72
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=8 obj/* $^ -o bin/$@-73
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=16 obj/* $^ -o bin/$@-74
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=32 obj/* $^ -o bin/$@-75
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=64 obj/* $^ -o bin/$@-76
	$(CC) $(CFLAGS) -DNSAMPLES=4096  -DKCLASSES=128 obj/* $^ -o bin/$@-77
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=2 obj/* $^ -o bin/$@-78
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=4 obj/* $^ -o bin/$@-79
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=8 obj/* $^ -o bin/$@-80
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=16 obj/* $^ -o bin/$@-81
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=32 obj/* $^ -o bin/$@-82
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=64 obj/* $^ -o bin/$@-83
	$(CC) $(CFLAGS) -DNSAMPLES=8192  -DKCLASSES=128 obj/* $^ -o bin/$@-84
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=2 obj/* $^ -o bin/$@-85
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=4 obj/* $^ -o bin/$@-86
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=8 obj/* $^ -o bin/$@-87
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=16 obj/* $^ -o bin/$@-88
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=32 obj/* $^ -o bin/$@-89
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=64 obj/* $^ -o bin/$@-90
	$(CC) $(CFLAGS) -DNSAMPLES=16384  -DKCLASSES=128 obj/* $^ -o bin/$@-91
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=2 obj/* $^ -o bin/$@-92
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=4 obj/* $^ -o bin/$@-93
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=8 obj/* $^ -o bin/$@-94
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=16 obj/* $^ -o bin/$@-95
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=32 obj/* $^ -o bin/$@-96
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=64 obj/* $^ -o bin/$@-97
	$(CC) $(CFLAGS) -DNSAMPLES=32768  -DKCLASSES=128 obj/* $^ -o bin/$@-98
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=2 obj/* $^ -o bin/$@-99
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=4 obj/* $^ -o bin/$@-100
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=8 obj/* $^ -o bin/$@-101
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=16 obj/* $^ -o bin/$@-102
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=32 obj/* $^ -o bin/$@-103
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=64 obj/* $^ -o bin/$@-104
	$(CC) $(CFLAGS) -DNSAMPLES=65536  -DKCLASSES=128 obj/* $^ -o bin/$@-105

clean:
	rm -rf bin
	rm -rf obj

test:
	bash test/run_tests.sh