#include "cusz.h"

#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"
#include "utils/io.hh"
#include "utils/print_gpu.hh"

extern "C" {
void** compress(float* deviceInputPtr, uint8_t* deviceCompressedPtr, int fileSize, float errorBound, size_t* compressedLen)
{
    auto len = fileSize;

    // cusz_header header;
    cusz_header* headerPtr = (cusz_header*)malloc(sizeof(cusz_header));
    uint8_t*    exposed_compressed;
    // size_t      compressed_len;

    float *d_uncompressed;
    
    d_uncompressed = deviceInputPtr;

    /* code snippet for looking at the device array easily */
    auto peek_devdata = [](float* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const float i) { printf("%f\t", i); });
        printf("\n");
    };

    /* a casual peek */
    printf("peeking uncompressed data, 20 elements\n");
    peek_devdata(d_uncompressed, 20);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // using default
    // cusz_framework* framework = cusz_default_framework();
    // alternatively
    cusz_framework* framework = new cusz_custom_framework{
        .pipeline     = Auto,
        .predictor    = cusz_custom_predictor{.type = LorenzoI},
        .quantization = cusz_custom_quantization{.radius = 512},
        .codec        = cusz_custom_codec{.type = Huffman}};

    cusz_compressor* comp       = cusz_create(framework, FP32);
    cusz_config*     config     = new cusz_config{.eb = errorBound, .mode = Rel};
    cusz_len         uncomp_len = cusz_len{len, 1, 1, 1};
    cusz_len         decomp_len = uncomp_len;

    cusz::TimeRecord compress_timerecord;

    {
        cusz_compress(
            comp, config, d_uncompressed, uncomp_len, &exposed_compressed, compressedLen, headerPtr,
            (void*)&compress_timerecord, stream);

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_compression(&compress_timerecord, len * sizeof(float), *compressedLen);

        /* verify header */
        printf("header.%-*s : %x\n", 12, "(addr)", headerPtr);
        printf("header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", headerPtr->x, headerPtr->y, headerPtr->z);
        printf("header.%-*s : %lu\n", 12, "filesize", ConfigHelper::get_filesize(headerPtr));
    }

    /* If needed, User should perform a memcopy to transfer `exposed_compressed` before `compressor` is destroyed. */
    // cudaMalloc(&compressed, compressed_len);
    cudaMemcpy(deviceCompressedPtr, exposed_compressed, *compressedLen, cudaMemcpyDeviceToDevice);

    void** res = (void**)malloc(sizeof(void*) * 3);
    res[0] = (void*) headerPtr;
    res[1] = (void*) comp;
    res[2] = (void*) stream;

    return res;
}

void decompress(uint8_t* deviceCompressedPtr, float* deviceOutputPtr, int fileSize, float errorBound, size_t* compressedLen, void** ptrs)
{
    auto len = fileSize;

    cusz_header* headerPtr = (cusz_header*)ptrs[0];
    // uint8_t*    exposed_compressed;
    // size_t      compressed_len;

    float *d_decompressed;
    
    d_decompressed = deviceOutputPtr;

    auto peek_devdata = [](float* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const float i) { printf("%f\t", i); });
        printf("\n");
    };

    cudaStream_t stream = (cudaStream_t)ptrs[2];

    cusz_compressor* comp       = (cusz_compressor*)ptrs[1];
    cusz_len         uncomp_len = cusz_len{len, 1, 1, 1};
    cusz_len         decomp_len = uncomp_len;

    cusz::TimeRecord decompress_timerecord;

    {
        cusz_decompress(
            comp, headerPtr, deviceCompressedPtr, *compressedLen, d_decompressed, decomp_len,
            (void*)&decompress_timerecord, stream);

        cusz::TimeRecordViewer::view_decompression(&decompress_timerecord, len * sizeof(float));
    }

    /* a casual peek */
    printf("peeking decompressed data, 20 elements\n");
    peek_devdata(d_decompressed, 20);

    free(headerPtr);
    cusz_release(comp);
    cudaStreamDestroy(stream);
}
}