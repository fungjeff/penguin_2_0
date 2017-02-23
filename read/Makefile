IDIR=../include
ODIR=../obj
SDIR=../src
LIB=-lmgl

CC=nvcc -arch=sm_35
CFLAGS=-I$(IDIR)

_DEPS = post_global.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = post_global.o post.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: post

post: $(OBJ)
	$(CC) -o $@ $^  $(LIB)

$(ODIR)/post.o: post.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS) $(LIB)

$(ODIR)/post_global.o: $(SDIR)/shared/post_global.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

clean:
	rm -f post *~ *linkinfo
	rm -f $(OBJ)
	rm -f $(IDIR)/*~
	rm -f $(ODIR)/*linkinfo
	rm -f $(SDIR)/shared/*~
