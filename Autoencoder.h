#pragma once
#include <vector>

struct AutoencoderTrainingOptions
{
	unsigned int epochs;
	float regularisation;
	float initialWeightNoiseScale;
};

constexpr static AutoencoderTrainingOptions k_DefaultTrainingOption
{
	1000,
	0.01f,
	0.01f,
};

struct Autoencoder;

Autoencoder* InitializeAutoencoder(int layers, int indims, int middims);
void FinalizeAutoencoder(Autoencoder* ae);
void Encode(const Autoencoder* ae, const float* input, float* output);
void Decode(const Autoencoder* ae, const float* input, float* output);
double TrainAutoencoder(Autoencoder* ae, const std::vector<float*>& data, const AutoencoderTrainingOptions& options = k_DefaultTrainingOption);
bool SaveAutoencoder(const Autoencoder* ae, const char* path);
Autoencoder* LoadAutoencoder(const char* path);
