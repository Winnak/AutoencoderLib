#pragma once

struct AutoencoderTrainingOptions
{
	unsigned int epochs;
	double regularisation;
	double initialWeightNoiseScale;
	double adamLearningRate;
	double beta1;
	double beta2;
	unsigned int printInterval;
	bool useLowestLoss;
};

constexpr static AutoencoderTrainingOptions k_DefaultTrainingOption
{
	1000,
	0.01,
	0.001,
	0.01,
	0.9,
	0.999,
	100,
	true,
};

struct Autoencoder;

Autoencoder* InitializeAutoencoder(int layers, int indims, int middims);
void FinalizeAutoencoder(Autoencoder* ae);
void Encode(const Autoencoder* ae, const float* input, float* output);
void Decode(const Autoencoder* ae, const float* input, float* output);
unsigned int GetDecodedDimension(const Autoencoder* ae);
unsigned int GetEncodedDimension(const Autoencoder* ae);
unsigned int GetLayerCount(const Autoencoder* ae);
double TrainAutoencoder(Autoencoder* ae, const float** data, unsigned int dataCount, const AutoencoderTrainingOptions& options = k_DefaultTrainingOption);
bool SaveAutoencoder(const Autoencoder* ae, const char* path);
Autoencoder* LoadAutoencoder(const char* path);
