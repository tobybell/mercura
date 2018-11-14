OBJ=body.o vec3.o quat.o eoe.o

all: mercura interactive

mercura: simulator.o $(OBJ)
	gcc -o $@ $^

interactive: interactive.o $(OBJ)
	gcc -o $@ $^

%.o: %.c
	gcc -o $@ -c $^

clean:
	rm -rf mercura *.o
