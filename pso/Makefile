OBJ=body.o vec3.o policy.o sim.o eoe.o common.o
CFLAGS=

all: pso test

test: test.o $(OBJ)
	gcc $(CFLAGS) -o $@ $^

pso: pso.o $(OBJ)
	gcc $(CFLAGS) -o $@ $^

%.o: %.c
	gcc $(CFLAGS) -o $@ -c $^

clean:
	rm -f pso *.o
