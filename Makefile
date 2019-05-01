Objects = packet.o imu_data_decode.o
Cfiles = packet.c imu_data_decode.c
Cflags = -fPIC

libtotal.so:$(Objects)
	gcc -fPIC -shared $(Objects) -o $@
$(objects):%.o:%.c packet.h
	gcc -c $(Cflags) $< -o $@

.PHONY:clean
clean:
	rm -f *.o *.so
#gcc -fPIC -c imu_data_decode.c
#gcc -fPIC -c packet.c
#packet.o:	
#	gcc -fPIC -c packet.c


