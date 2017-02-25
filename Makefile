IDIR=./include
ODIR=./obj
SDIR=./src

CC=nvcc -O3 -arch=sm_35
CFLAGS=-I$(IDIR)

_DEPS = variable_types.h global.h disk_profile.h output.h timestep.h ppm.h planet.h orbital_advection.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = variable_types.o global.o disk_profile.o output.o timestep.o ppm.o planet.o orbital_advection.o main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: penguin

penguin: $(OBJ)
	$(CC) -o $@ $^

$(ODIR)/main.o: main.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/variable_types.o: $(SDIR)/shared/variable_types.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/global.o: $(SDIR)/shared/global.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/disk_profile.o: $(SDIR)/disk_profile/disk_profile.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/ppm.o: $(SDIR)/ppm/ppm.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/timestep.o: $(SDIR)/timestep/timestep.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/output.o: $(SDIR)/output/output.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/planet.o: $(SDIR)/planet/planet.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/orbital_advection.o: $(SDIR)/orbital_advection/orbital_advection.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

clean:
	rm -f penguin *~ *linkinfo
	rm -f $(ODIR)/*.o $(ODIR)/*linkinfo
	rm -f $(IDIR)/*~
	rm -f $(SDIR)/shared/*~
	rm -f $(SDIR)/disk_profile/*~
	rm -f $(SDIR)/ppm/*~
	rm -f $(SDIR)/timestep/*~
	rm -f $(SDIR)/output/*~
	rm -f $(SDIR)/orbital_advection/*~
