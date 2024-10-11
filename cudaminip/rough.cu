#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ctype.h>
#include <string.h>

typedef unsigned short dt;

typedef struct PGMImage {
    char pgmType[3];
    dt* data;
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
} PGMImage;


// CUDA kernel for the Laplacian filter
__global__ void laplacianFilter(int *inputImage, int *outputImage, int width, int height, int max) 
{
    // Calculate the pixel coordinates of the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image boundaries
    if (x < width && y < height)
    {
        // Calculate the index of the current pixel
        int pixelIndex = y * width + x;

        // Apply the Laplacian filter to the pixel and its neighbors
        int filteredPixel = 5 * inputImage[pixelIndex]
                          - inputImage[pixelIndex - 1]
                          - inputImage[pixelIndex + 1]
                          - inputImage[pixelIndex - width]
                          - inputImage[pixelIndex + width];

        // Set the filtered pixel value in the output image
        if(filteredPixel < 0)
            filteredPixel = 0;
        else if(filteredPixel > max)
            filteredPixel = max;
        outputImage[pixelIndex] = filteredPixel;
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

    dim3 dimGrid(ceil(pgm->width/16.0), ceil(pgm->height/16.0));
    dim3 dimBlock(16, 16);

    laplacianFilter<<<dimGrid, dimBlock>>>(d_in, d_out, pgm->width, pgm->height, pgm->maxValue);

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

    FILE* out_pgmfile = fopen("lena_laplacian_out.pgm", "wb+");
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

