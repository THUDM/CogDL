CC?=gcc # Set compiler if CC is not set
CFLAGS= -fopenmp -fPIC -O3 -D NDEBUG -Wall -Werror

all: librwalk.so

librwalk.so: rwalk.o
	$(CC) $(CFLAGS) -shared -Wl,-soname,librwalk.so -o librwalk.so rwalk.o
	rm rwalk.o

rwalk.o: rwalk.c
	$(CC) -c $(CFLAGS) rwalk.c -o rwalk.o

clean :
	rm -rf librwalk.so rwalk.o __pycache__
