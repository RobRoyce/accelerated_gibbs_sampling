CC=nvcc
OPT=0
DLTO=
OBJ=obj
BIN=bin
TEST=test
NSAMPLES=131072
KCLASSES=16
NITERS=500

CFLAGS=--gpu-architecture=sm_75 -O$(OPT) $(DLTO) -lm -lcurand -rdc=true
DFLAGS=-DNSAMPLES=$(NSAMPLES) -DKCLASSES=$(KCLASSES) -DNITERS=$(NITERS)


DEPS=src/distrs.cu src/utils.cu src/gmm.cu src/gmm_gibbs.cu
INIT := init
MODULES := distrs utils gmm gmm_gibbs
TESTS := cont_pdf_test cont_gof_test int_test


CSV=$(TEST)/results.csv
EXEC_FILES := $(shell find $(BIN)/int* 2> /dev/null)


.PHONY : clean test


all: $(INIT) $(TESTS)

profile: init
	filenum=0
	for n in 1024 2048 4096 8192 16384 32768 65536 131072 262144 ; do \
		for k in 2 4 8 16 32 64 128 ; do \
		  	make int_test NSAMPLES=$$n KCLASSES=$$k OPT=1 DLTO=-dlto; \
		done \
  	done

exec:
	echo "N,K,Time(usec)" > $(CSV)
	for file in $(EXEC_FILES) ; do \
  		./$$file >> $(CSV); \
  		./$$file >> $(CSV); \
  		./$$file >> $(CSV); \
  		./$$file >> $(CSV); \
  		./$$file >> $(CSV); \
	done

debug: CFLAGS += -g -G
debug: all

init:
	mkdir -p $(OBJ) $(BIN)

modules: $(MODULES)
	$(CC) --device-c $(CFLAGS) $(DFLAGS) $^ -odir ./$(OBJ)/

cont_pdf_test: test/cont_pdf_test.cu
	#$(CC) --link $(CFLAGS) $(DFLAGS) $^ $(DEPS) -o $(BIN)/$@

cont_gof_test: test/cont_gof_test.cu
	#$(CC) --link $(CFLAGS) $(DFLAGS) $^ $(DEPS) -o $(BIN)/$@

int_test: test/int_test.cu
	$(CC) --link $(CFLAGS) $(DFLAGS) $^ $(DEPS) -o $(BIN)/$@-$(NSAMPLES)-$(KCLASSES)

clean:
	rm -rf bin
	rm -rf obj

test:
	cd bin && ./cont_gof_test && ./int_test && ./cont_pdf_test
