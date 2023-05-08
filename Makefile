all: capi.cu
	nvcc -arch=sm_80 -Xcompiler -fPIC -shared --extended-lambda --expt-relaxed-constexpr -I/home/bozhan/anaconda3/envs/dev/include/python3.10 -L/home/bozhan/repo/cusz-sc/build -lcusz -I/home/bozhan/repo/cusz-sc/include capi.cu -o capi.so

clean:
	rm -rf capi.so