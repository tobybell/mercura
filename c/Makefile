build/mercura: build simulator.o body.o vec3.o
	gcc -o $@ simulator.o body.o vec3.o

%.o: %.c
	gcc -o $@ -c $^

build:
	mkdir $@

clean:
	rm -f mercura *.o
