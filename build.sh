swgcc -mslave -O3 -msimd -funroll-loops slave.c -c -o slave.o #-funroll-loops
swgcc -faddress_align=128 -mhost -msimd -O3 master.c -c -o master.o
swg++ -faddress_align=128 -mhost -msimd -O3 main.cpp -c -o main.o
swg++ -mhybrid -static main.o master.o slave.o -o exe 