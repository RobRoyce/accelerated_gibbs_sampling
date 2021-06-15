CC=nvcc
OPT=0
DLTO=
CFLAGS=--gpu-architecture=sm_75 -O$(OPT) $(DLTO) -lm -lcurand -rdc=true
DFLAGS=-DNSAMPLES=$(NSAMPLES) -DKCLASSES=$(KCLASSES)
OBJ=obj
BIN=bin
TEST=test
CSV=$(TEST)/results.csv

INIT := init
MODULES := distrs utils gmm gmm_gibbs
TESTS := cont_pdf_test cont_gof_test int_test

EXEC_FILES := $(shell find $(BIN)/int*)
NSAMPLES=262144
KCLASSES=64

.PHONY : clean test
all: $(INIT) modules $(TESTS)

profile: init modules
	filenum=0
	for n in 16384 32768 65536 131072 262144 ; do \
		for k in 2 4 8 16 32 64 128 ; do \
		  	make int_test NSAMPLES=$$n KCLASSES=$$k OPT=1 DLTO=-dlto; \
		done \
  	done

exec:
	echo "N,K,Time(usec)" > $(CSV)
	for file in $(EXEC_FILES) ; do \
  		./$$file >> $(CSV); \
	done

debug: CFLAGS += -g -G
debug: all

init:
	mkdir -p $(OBJ) $(BIN)

modules: src/distrs.cu src/utils.cu src/gmm.cu src/gmm_gibbs.cu
	$(CC) --device-c $(CFLAGS) $(DFLAGS) $^ -odir ./$(OBJ)/

cont_pdf_test: test/cont_pdf_test.cu
	$(CC) --link $(CFLAGS) $(DFLAGS) $^ obj/* -o $(BIN)/$@

cont_gof_test: test/cont_gof_test.cu
	$(CC) --link $(CFLAGS) $(DFLAGS) $^ obj/* -o $(BIN)/$@

int_test: test/int_test.cu
	$(CC) --device-c $(CFLAGS) $(DFLAGS) src/distrs.cu src/utils.cu src/gmm.cu src/gmm_gibbs.cu -odir ./obj/
	$(CC) --link $(CFLAGS) $(DFLAGS) $^ obj/* -o $(BIN)/$@-$(NSAMPLES)-$(KCLASSES)

clean:
	rm -rf bin
	rm -rf obj

test:
	cd bin && ./cont_gof_test && ./int_test && ./cont_pdf_test
