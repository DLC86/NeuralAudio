#include "NeuralAudioCAPI/NeuralAudioCApi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "Usage: pnam_render <model.pnam>\n"); return 1; }
    NeuralModelLoader* loader = CreateLoader();
    SetDefaultMaxAudioBufferSize(loader, 128);
    NeuralModel* model = CreateModelFromFileUtf8(loader, argv[1]);
    if (!model || !IsParametric(model)) { DeleteModel(model); DeleteLoader(loader); return 2; }
    const size_t count = PnamGetControlCount(model);
    float* values = (float*)calloc(count, sizeof(float));
    if (!values) { DeleteModel(model); DeleteLoader(loader); return 3; }
    for (size_t i = 0; i < count; ++i) {
        float low, high;
        PnamGetControlRange(model, i, &low, &high, &values[i]);
        printf("%s: %g .. %g (default %g)\n", PnamGetControlName(model, i), low, high, values[i]);
        for (size_t p = 0; p < PnamGetPositionCount(model, i); ++p)
            printf("  %zu: %s\n", p, PnamGetPositionName(model, i, p));
    }
    PnamSetControls(model, values, count);
    PnamSetCalibrationLevelDBu(model, 12.0f);
    PnamSetOutputMode(model, PNAM_AUTO);
    printf("Requested mode: %d; effective mode: %d; sample rate: %g\n",
           PnamGetOutputMode(model), PnamGetEffectiveOutputMode(model), GetSampleRate(model));
    // Replace these buffers with the mono audio buffers provided by your hardware.
    // Apply GetRecommendedInputDBAdjustment() in the host if input calibration is enabled.
    // Process already applies the selected output mode; do not apply it a second time.
    float input[128] = {0}, output[128];
    Process(model, input, output, 128);
    free(values);
    DeleteModel(model);
    DeleteLoader(loader);
    return 0;
}
