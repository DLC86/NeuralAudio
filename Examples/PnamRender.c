#include "NeuralAudioCAPI/NeuralAudioCApi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void Require(int condition, const char* message) {
    if (!condition) { fprintf(stderr, "%s\n", message); exit(4); }
}

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
    Require(PnamSetControls(model, values, count) == 1, "Set controls failed");
    Require(PnamSetCalibrationLevelDBu(model, 12.0f) == 1, "Calibration failed");
    Require(PnamSetOutputMode(model, PNAM_AUTO) == 1, "Auto output mode failed");
    Require(PnamGetOutputMode(model) == PNAM_AUTO, "Requested mode was lost");
    Require(PnamSetOutputMode(model, 999) == 0 && PnamGetOutputMode(model) == PNAM_AUTO,
            "Invalid mode changed the selection");
    if (PnamHasLite(model)) {
        Require(PnamSetLite(model, true) == 1, "Lite selection failed");
        Require(PnamSetLite(model, false) == 1, "Full selection failed");
    }
    const size_t bytes = PnamGetMetadataJson(model, NULL, 0);
    Require(bytes > 1, "Metadata query failed");
    char small[2] = {'x', 'y'};
    Require(PnamGetMetadataJson(model, small, 2) == bytes && small[0] == 'x' && small[1] == 'y',
            "An undersized metadata buffer was overwritten");
    char* metadata = (char*)malloc(bytes);
    Require(metadata != NULL, "Metadata allocation failed");
    Require(PnamGetMetadataJson(model, metadata, bytes) == bytes && metadata[0] == '{' &&
            metadata[bytes - 1] == '\0', "Metadata retrieval failed");
    free(metadata);
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
