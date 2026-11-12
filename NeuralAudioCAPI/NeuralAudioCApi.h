#pragma once

#include <stddef.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef NEURALAUDIO_CAPI_BUILD
#define NA_EXTERN extern __declspec(dllexport)
#else
#define NA_EXTERN extern __declspec(dllimport)
#endif
#else
#define NA_EXTERN extern
#endif

typedef struct NeuralModel NeuralModel;
typedef struct NeuralModelLoader NeuralModelLoader;

NA_EXTERN NeuralModelLoader* CreateLoader();

NA_EXTERN void DeleteLoader(NeuralModelLoader* loader);

NA_EXTERN NeuralModel* CreateModelFromFile(NeuralModelLoader *loader, const wchar_t* modelPath);
NA_EXTERN NeuralModel* CreateModelFromFileUtf8(NeuralModelLoader *loader, const char* modelPath);

NA_EXTERN void DeleteModel(NeuralModel* model);

NA_EXTERN void SetLSTMLoadMode(NeuralModelLoader* loader, int loadMode);

NA_EXTERN void SetWaveNetLoadMode(NeuralModelLoader* loader, int loadMode);

NA_EXTERN void SetAudioInputLevelDBu(NeuralModelLoader* loader, float audioDBu);

NA_EXTERN void SetDefaultMaxAudioBufferSize(NeuralModelLoader* loader, int maxSize);

NA_EXTERN int GetLoadMode(NeuralModel* model);

NA_EXTERN bool IsStatic(NeuralModel* model);

NA_EXTERN void SetMaxAudioBufferSize(NeuralModel* model, int maxSize);

NA_EXTERN float GetRecommendedInputDBAdjustment(NeuralModel* model);

NA_EXTERN float GetRecommendedOutputDBAdjustment(NeuralModel* model);
// Output mode is Auto by default for all model types; the host applies the
// returned trim for static models, while PNAM applies it within Process().
NA_EXTERN int SetModelOutputMode(NeuralModel* model, int mode);
NA_EXTERN int GetModelOutputMode(NeuralModel* model);
NA_EXTERN int GetModelEffectiveOutputMode(NeuralModel* model);

NA_EXTERN float GetSampleRate(NeuralModel* model);

NA_EXTERN void Process(NeuralModel* model, float* input, float* output, size_t numSamples);

// PNAM additions. Call setters between Process calls on the same thread.
// Size changes/Prepare are offline operations. New setters return 1 on success, 0 on error.
enum PnamOutputMode { PNAM_RAW = 0, PNAM_CALIBRATED = 1, PNAM_NORMALIZED = 2, PNAM_AUTO = 3 };
NA_EXTERN bool IsParametric(NeuralModel* model);
NA_EXTERN size_t PnamGetControlCount(NeuralModel* model);
NA_EXTERN const char* PnamGetControlName(NeuralModel* model, size_t index);
NA_EXTERN int PnamGetControlRange(NeuralModel* model, size_t index, float* minimum, float* maximum, float* nominal);
NA_EXTERN size_t PnamGetPositionCount(NeuralModel* model, size_t index);
NA_EXTERN const char* PnamGetPositionName(NeuralModel* model, size_t index, size_t position);
// A displayed option index_sw maps to the original 0..N-1 neural switch value.
NA_EXTERN int PnamGetPositionValue(NeuralModel* model, size_t index, size_t position);
NA_EXTERN int PnamSetControls(NeuralModel* model, const float* values, size_t count);
// Developer-configurable, independent milliseconds; default 100/100, linear.
NA_EXTERN int PnamSetWeightUpdateIntervalMs(NeuralModel* model, float ms);
NA_EXTERN int PnamSetWeightSmoothingMs(NeuralModel* model, float ms);
NA_EXTERN int PnamSetOutputMode(NeuralModel* model, int mode);
NA_EXTERN int PnamGetOutputMode(NeuralModel* model);
NA_EXTERN int PnamGetEffectiveOutputMode(NeuralModel* model);
NA_EXTERN int PnamSetCalibrationLevelDBu(NeuralModel* model, float level);
NA_EXTERN int PnamSetLite(NeuralModel* model, bool lite);
NA_EXTERN bool PnamHasLite(NeuralModel* model);
// Returns required bytes including NUL. Pass NULL/0 to query; metadata is never truncated.
NA_EXTERN size_t PnamGetMetadataJson(NeuralModel* model, char* buffer, size_t capacity);

#ifdef __cplusplus
}
#endif
