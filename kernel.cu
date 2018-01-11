
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

using namespace std;

cudaError_t fractalAddCuda(int* out, const int* width, const int* height, dim3 gridSize, dim3 blockSize, int type, int* max_iter, double* zoom, double* pan, double* c, unsigned int size);

__global__ void mandelbrotKernel(int* out, int* width, int* height, int* max_iter, double* zoom, double* pan)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = py * *width + px;
	if (px >= *width || py >= *height) return;

	double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
	double y0 = (double)py / (double)*height * 2 - 1 - pan[1] / *zoom;
	double x = 0.0;
	double y = 0.0;
	double xtemp = 0;
	double ytemp = 0;
	double iter = 0;
			
	double p = sqrt((x - 1.0 / 4.0) * (x - 1.0 / 4.0) + y * y);

	if (x < p - 2.0 * (p * p) + 1.0 / 4.0) {
		while (x * x + y * y < 4.0 && iter < *max_iter) {
			xtemp = x * x - y * y + x0;
			ytemp = 2.0 * x * y + y0;
			if (x == xtemp && y == ytemp) {
				iter = *max_iter;
				break;
			}
			y = ytemp;
			x = xtemp;
			iter++;
		}
	}

	//double miter = *max_iter;
	int color = iter / *max_iter * 255.0;
	out[idx] = color;
}

__global__ void juliaKernel(int* out, int* width, int* height, int* max_iter, double* zoom, double* pan, double* c)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = py * *width + px;
	if (px >= *width || py >= *height) return;

	double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
	double y0 = (double)py / (double)*height * 2 - 1 - pan[1] / *zoom;
	double xtemp = 0;
	int iter = 0;

	while (x0 * x0 + y0 * y0 < 4 && iter < *max_iter) {
		xtemp = x0 * x0 - y0 * y0 + x0;
		y0 = 2.0 * x0 * y0 + y0 + *c;
		x0 = xtemp + *c;
		iter++;
	}

	int color = (sin(iter - logf(logf(iter)) / logf(2) / *max_iter) + 1) / 2 * 255;
	out[idx] = color;
}

__global__ void burningShipKernel(int* out, int* width, int* height, int* max_iter, double* zoom, double* pan)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = py * *width + px;
	if (px >= *width || py >= *height) return;

	double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
	double y0 = (double)py / (double)*height * 2 - 1.5 - pan[1] / *zoom;
	double x = 0.0;
	double y = 0.0;
	double xtemp = 0;
	int iter = 0;

	while (x * x + y * y < 4 && iter < *max_iter) {
		xtemp = x * x - y * y + x0;
		y = abs(2.0 * x * y + y0);
		x = abs(xtemp);
		iter++;
	}

	double color = iter % 255;
	out[idx] = color;
}

int* cpuMandelbrotKernel(int* width, int* height, int* max_iter, double* zoom, double* pan)
{
	int* out = new int[*width * *height];
	for (int px = 0; px < *width; px++) {
		for (int py = 0; py < *height; py++) {
			double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
			double y0 = (double)py / (double)*height * 2 - 1 - pan[1] / *zoom;
			double x = 0.0;
			double y = 0.0;
			double xtemp = 0;
			double ytemp = 0;
			double iter = 0;

			double p = sqrt((x - 1.0 / 4.0) * (x - 1.0 / 4.0) + y * y);

			if (x < p - 2.0 * (p * p) + 1.0 / 4.0) {
				while (x * x + y * y < 4.0 && iter < *max_iter) {
					xtemp = x * x - y * y + x0;
					ytemp = 2.0 * x * y + y0;
					if (x == xtemp && y == ytemp) {
						iter = *max_iter;
						break;
					}
					y = ytemp;
					x = xtemp;
					iter++;
				}
			}

			//double miter = *max_iter;
			int color = iter / *max_iter * 255.0;
			out[py * *width + px] = color;
		}
	}
	return out;
}

int* cpuJuliaKernel(int* width, int* height, int* max_iter, double* zoom, double* pan, double* c)
{
	int* out = new int[*width * *height];
	for (int px = 0; px < *width; px++) {
		for (int py = 0; py < *height; py++) {
			double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
			double y0 = (double)py / (double)*height * 2 - 1 - pan[1] / *zoom;
			double xtemp = 0;
			int iter = 0;

			while (x0 * x0 + y0 * y0 < 4 && iter < *max_iter) {
				xtemp = x0 * x0 - y0 * y0 + x0;
				y0 = 2.0 * x0 * y0 + y0 + *c;
				x0 = xtemp + *c;
				iter++;
			}

			int color = (sin(iter - logf(logf(iter)) / logf(2) / *max_iter) + 1) / 2 * 255;
			out[py * *width + px] = color;
		}
	}
	return out;
}

int* cpuBurningShipKernel(int* width, int* height, int* max_iter, double* zoom, double* pan)
{
	int* out = new int[*width * *height];
	for (int px = 0; px < *width; px++) {
		for (int py = 0; py < *height; py++) {
			double x0 = (double)px / (double)*width * 3.5 - 2.5 - pan[0] / *zoom;
			double y0 = (double)py / (double)*height * 2 - 1.5 - pan[1] / *zoom;
			double x = 0.0;
			double y = 0.0;
			double xtemp = 0;
			int iter = 0;

			while (x * x + y * y < 4 && iter < *max_iter) {
				xtemp = x * x - y * y + x0;
				y = abs(2.0 * x * y + y0);
				x = abs(xtemp);
				iter++;
			}

			double color = iter % 255;
			out[py * *width + px] = color;
		}
	}
	return out;
}

int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

int main(int argc, char *argv[])
{
	//argc = 12;//cfractalplus 0 3840 2160 0 0 1000 1 1 16 16 0
	//argv = new char*[12]{ "cfractalplus", "2", "1920", "1080", "0", "0", "1000", "1", "1", "16", "16", "1" };
	if (argc < 12) {
		std::cout << "invalid arguments\n";
		std::cout << "cfractalplus [type (0: mandelbrot, 1: julia set, 2: burning ship)] [width] [height] [pan x] [pan y] [max iteration] [zoom] [c] [block size x] [block size y] [cpu? 0: no, 1: yes]";
	}
	int* c = new int[atoi(argv[2]) * atoi(argv[3])];
	if (atoi(argv[11]) == 0) {
		std::cout << "Starting CUDA operations\n";
		dim3 blockDim(atoi(argv[9]), atoi(argv[10]));
		dim3 gridDim(atoi(argv[2]) / blockDim.x, atoi(argv[3]) / blockDim.y);
		cudaError_t cudaStatus = fractalAddCuda(c, new int[1]{ atoi(argv[2]) }, new int[1]{ atoi(argv[3]) }, gridDim, blockDim, atoi(argv[1]), new int[1]{ atoi(argv[6]) }, (double*)(new double[1]{ atof(argv[7]) }), (double*)(new double[1]{ atof(argv[4]) }), ((double*)(new double[1]{ atof(argv[5]) }), (double*)(new double[1]{ atof(argv[8]) })), atoi(argv[2]) * atoi(argv[3]));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		std::cout << "Finished CUDA operations\n";
		std::cout << "Writing to file\n";
		string filename = "";
		std::cout << "Filename (needs .ppm at end)? ";
		std::cin >> filename;
		ofstream fs(filename);
		fs << "P3\n" << atoi(argv[2]) << "\n" << atoi(argv[3]) << "\n255\n";
		for (int i = 0; i < atoi(argv[2]) * atoi(argv[3]); i++) {
			fs << c[i] << " " << c[i] << " " << c[i] << "\n";
		}
		fs.close();
		std::cout << "Finished";
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
	else {
		std::cout << "Starting CPU operations\n";
		switch (atoi(argv[1])) {
			case 0: c = cpuMandelbrotKernel(new int[1]{ atoi(argv[2]) }, new int[1]{ atoi(argv[3]) }, new int[1]{ atoi(argv[6]) }, (double*)(new double[1]{ atof(argv[7]) }), (double*)(new double[1]{ atof(argv[4]) })); break;
			case 1: c = cpuJuliaKernel(new int[1]{ atoi(argv[2]) }, new int[1]{ atoi(argv[3]) }, new int[1]{ atoi(argv[6]) }, (double*)(new double[1]{ atof(argv[7]) }), (double*)(new double[1]{ atof(argv[4]) }), (double*)(new double[1]{ atof(argv[8]) })); break;
			case 2: c = cpuBurningShipKernel(new int[1]{ atoi(argv[2]) }, new int[1]{ atoi(argv[3]) }, new int[1]{ atoi(argv[6]) }, (double*)(new double[1]{ atof(argv[7]) }), (double*)(new double[1]{ atof(argv[4]) })); break;
		}
		std::cout << "Finished CPU operations\n";
		std::cout << "Writing to file\n";
		string filename = "";
		std::cout << "Filename (needs .ppm at end)? ";
		std::cin >> filename;
		ofstream fs(filename);
		fs << "P3\n" << atoi(argv[2]) << "\n" << atoi(argv[3]) << "\n255\n";
		for (int i = 0; i < atoi(argv[2]) * atoi(argv[3]); i++) {
			fs << c[i] << " " << c[i] << " " << c[i] << "\n";
		}
		fs.close();
		std::cout << "Finished";
	}
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t fractalAddCuda(int* out, const int* width, const int* height, dim3 gridSize, dim3 blockSize, int type, int* max_iter, double* zoom, double* pan, double* c, unsigned int size)
{
	int *dev_width = 0;
	int *dev_height = 0;
	int *dev_out = 0;
	int *dev_iter = 0;
	double *dev_zoom = 0;
	double *dev_pan = 0;
	double *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_width, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_height, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_iter, 1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_zoom, 1 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pan, 2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	if (type == 1) {
		cudaStatus = cudaMalloc((void**)&dev_c, 1 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_width, width, 1 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpya failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_height, height, 1 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyb failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_iter, max_iter, 1 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_zoom, zoom, 1 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyd failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pan, pan, 2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpye failed!");
		goto Error;
	}

	if (type == 1)
	{
		cudaStatus = cudaMemcpy(dev_c, c, 1 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpyf failed!");
			goto Error;
		}
	}

	switch (type) {
		case 0: mandelbrotKernel<<<gridSize, blockSize>>>(dev_out, dev_width, dev_height, dev_iter, dev_zoom, dev_pan);
			break;
		case 1: juliaKernel<<<gridSize, blockSize>>>(dev_out, dev_width, dev_height, dev_iter, dev_zoom, dev_pan, dev_c);
			break;
		case 2: burningShipKernel<<<gridSize, blockSize>>>(dev_out, dev_width, dev_height, dev_iter, dev_zoom, dev_pan);
			break;
	}

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(&dev_out);
    cudaFree(dev_width);
    cudaFree(dev_height);
	cudaFree(dev_zoom);
	cudaFree(dev_pan);
	cudaFree(dev_iter);
	if (type == 1) cudaFree(dev_c);
    return cudaStatus;
}
