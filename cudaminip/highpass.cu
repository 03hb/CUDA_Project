#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define FILTER_SIZE 3

typedef unsigned short dt;

typedef struct PGMImage {
    char pgmType[3];
    dt* data;
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
} PGMImage;


// CUDA kernel for the High-Pass filter
__global__ void highpassFilter(int *inputImage, int *outputImage, int width, int height, int max) 
{
    // Calculate the pixel coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if (x < width && y < height) {
        // Calculate the index of the current pixel
        int pixelIndex = y * width + x;

        // Initialize the sum and the normalization factor
        int sum = 0;
        int count = 0;

        // Iterate over the filter window
        for (int fy = -FILTER_SIZE/2; fy <= FILTER_SIZE/2; fy++) {
            for (int fx = -FILTER_SIZE/2; fx <= FILTER_SIZE/2; fx++) {
                // Calculate the coordinates of the neighboring pixel
                int nx = x + fx;
                int ny = y + fy;

                // Check if the neighboring pixel is within the image boundaries
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Calculate the index of the neighboring pixel
                    int neighborIndex = ny * width + nx;

                    // Check if the neighboring pixel is not the center pixel
                    if (nx != x || ny != y) {
                        // Accumulate the sum and the count
                        sum += inputImage[neighborIndex];
                        count++;
                    }
                }
            }
        }

        // Calculate the filtered pixel value as the difference between the center pixel and the average of its neighbors
        int filteredValue = inputImage[pixelIndex] - (sum / count);

        if (filteredValue < 0)
            filteredValue = 0;
        else if(filteredValue > max)
            filteredValue = max;

        // Set the filtered pixel value in the output image
        outputImage[pixelIndex] = filteredValue;
    }
}

void ignoreComments(FILE* fp)
{
    int ch;
    char line[100];

    while ((ch = fgetc(fp)) != EOF && isspace(ch));

    if (ch == '#') 
    {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    }
    else
        fseek(fp, -1, SEEK_CUR);
}

int main()
{
    PGMImage* pgm = (PGMImage*)malloc(sizeof(PGMImage));
    const char* ipfile = "lena.ascii.pgm";
        
    FILE* pgmfile = fopen(ipfile, "rb");

    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return false;
    }

    ignoreComments(pgmfile);
    fscanf(pgmfile, "%s", pgm->pgmType);

    ignoreComments(pgmfile);

    fscanf(pgmfile, "%d %d",&(pgm->width), &(pgm->height));

    ignoreComments(pgmfile);

    fscanf(pgmfile, "%d", &(pgm->maxValue));
    ignoreComments(pgmfile);

    fgetc(pgmfile);
    pgm->data = (dt *)malloc(pgm->width*pgm->height*sizeof(dt));
    for (int i = 0; i < pgm->width * pgm->height; i++) {
        int pixel;
        fscanf(pgmfile, "%d", &pixel);
        // Clamp pixel values to [0, maxval]
        pixel = (pixel < 0) ? 0 : ((pixel > pgm->maxValue) ? pgm->maxValue : pixel);
        pgm->data[i] = (dt)((double)pixel / pgm->maxValue * 255);
    }

    fclose(pgmfile);

    int *d_in, *d_out, *in, *out;
    int size = pgm->width*pgm->height*sizeof(int);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    in = (int*)malloc(size);
    out = (int *)malloc(size);
    for(int i=0;i<pgm->width*pgm->height;i++)
        in[i] = pgm->data[i];
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil((float) pgm->width / BLOCK_SIZE), ceil((float) pgm->height / BLOCK_SIZE));

    highpassFilter<<<gridSize, blockSize>>>(d_in, d_out, pgm->width, pgm->height, pgm->maxValue);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    int **out_pix = (int **)malloc(pgm->width*sizeof(int *));
    int i,j,k=0;     
    for(i=0;i<pgm->height;i++)
    {
        out_pix[i] = (int *)malloc(pgm->width*sizeof(int));
        for(j=0;j<pgm->width;j++)
        {
            out_pix[i][j] = (int)out[k++];
        }
    }

    FILE* out_pgmfile = fopen("lena_highpass_out.pgm", "wb+");
    fprintf(out_pgmfile, "P2\n");
    fprintf(out_pgmfile, "%d %d\n", pgm->height, pgm->width);
    fprintf(out_pgmfile, "%d\n", pgm->maxValue);
    for(int i=0;i<pgm->height;i++)
    {
        for(int j=0; j<pgm->width; j++)
        {
            fprintf(out_pgmfile, "%d ",out_pix[i][j]);
        }
        fprintf(out_pgmfile, "\n");
    }

    fclose(out_pgmfile);
    
    return 0; 

}