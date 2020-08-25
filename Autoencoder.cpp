#include <shark/Models/LinearModel.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/GradientDescent/Adam.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h> 

#include <vector>

#include "Autoencoder.h"

typedef unsigned int uint32;

struct Autoencoder
{
	uint32 layers;
	uint32 inDims;
	uint32 midDims;

	shark::ConcatenatedModel<shark::FloatVector> encoder;
	shark::ConcatenatedModel<shark::FloatVector> decoder;
	shark::ConcatenatedModel<shark::FloatVector> model;
};


static void SetupLayers(Autoencoder * ae)
{
	//We use a dense linear model with rectifier activations
	typedef shark::LinearModel<shark::FloatVector, shark::RectifierNeuron> DenseLayer;

	int step = (ae->inDims - ae->midDims) / (ae->layers);
	for (uint32 i = 1; i < ae->layers; i++)
	{
		ae->encoder.add(new DenseLayer(ae->inDims - (i - 1) * step, ae->inDims - i * step), true);
		ae->decoder.add(new DenseLayer(ae->midDims + (i - 1) * step, ae->midDims + i * step), true);

	}
	ae->encoder.add(new DenseLayer(ae->inDims - (ae->layers - 1) * step, ae->midDims), true);
	ae->decoder.add(new DenseLayer(ae->midDims + (ae->layers - 1) * step, ae->inDims), true);
	ae->model = ae->encoder >> ae->decoder;
}

Autoencoder* InitializeAutoencoder(int layers, int indims, int middims)
{
	Autoencoder* ae = new Autoencoder();
	ae->layers = layers;
	ae->inDims = indims;
	ae->midDims = middims;

	SetupLayers(ae);

	return ae;
}

void FinalizeAutoencoder(Autoencoder* ae)
{
	ae->layers = 0;
	ae->inDims = 0;
	ae->midDims = 0;

	delete ae;
	ae = nullptr;
}

void Encode(const Autoencoder* ae, const float* input, float* output)
{
	shark::FloatVector fv(input, input + ae->inDims);
	const auto b = ae->encoder(fv);
	std::move(b.cbegin(), b.cend(), output);
}

void Decode(const Autoencoder* ae, const float* input, float* output)
{
	shark::FloatVector fv(input, input + ae->midDims);
	const auto b = ae->decoder(fv);
	std::move(b.cbegin(), b.cend(), output);
}

double TrainAutoencoder(Autoencoder* ae, const float* data, unsigned int dataCount, const AutoencoderTrainingOptions& options = k_DefaultTrainingOption);
{
	std::vector<shark::FloatVector> fvData;
	for (size_t i = 0; i < dataCount; i++)
	{
		const shark::FloatVector fv(data[i], data[i] + ae->inDims);
		fvData.push_back(fv);
	}
	shark::UnlabeledData<shark::FloatVector> ds = shark::createDataFromRange(fvData);
	
	shark::LabeledData<shark::FloatVector, shark::FloatVector> trainSet(ds, ds);
	shark::SquaredLoss<shark::FloatVector> loss;
	shark::ErrorFunction<shark::FloatVector> error(trainSet, &ae->model, &loss, true);
	shark::TwoNormRegularizer<shark::FloatVector> regularizer(error.numberOfVariables());
	error.setRegularizer(options.regularisation, &regularizer);

	shark::initRandomNormal(ae->model, options.initialWeightNoiseScale);

	shark::Adam<shark::FloatVector> optimizer;
	error.init();
	optimizer.init(error);
	for (uint32 i = 0; i != options.epochs; ++i)
	{
		optimizer.step(error);
		if (i % 100 == 0)
		{
			std::cout << i << " " << optimizer.solution().value << std::endl;
		}
	}

	ae->model.setParameterVector(optimizer.solution().point);
	return optimizer.solution().value;
}

bool SaveAutoencoder(const Autoencoder* ae, const char* path)
{
	std::ofstream ofs(path);
	if (!ofs.is_open())
	{
		return false;
	}

	shark::TextOutArchive oa(ofs);
	oa << ae->layers << ae->inDims << ae->midDims;
	ae->encoder.save(oa, 1);
	ae->decoder.save(oa, 1);
	ofs.close();

	return true;
}

Autoencoder* LoadAutoencoder(const char* path)
{
	std::ifstream ifs(path);
	if (!ifs.is_open())
	{
		return nullptr;
	}

	Autoencoder* outAe = new Autoencoder();

	shark::TextInArchive ia(ifs);
	ia >> outAe->layers;
	ia >> outAe->inDims;
	ia >> outAe->midDims;
	SetupLayers(outAe);
	outAe->encoder.load(ia, 1);
	outAe->decoder.load(ia, 1);
	ifs.close();

	return outAe;
}

unsigned int GetDecodedDimension(const Autoencoder* ae)
{
	return ae->inDims;
}

unsigned int GetEncodedDimension(const Autoencoder* ae)
{
	return ae->midDims;
}

unsigned int GetLayerCount(const Autoencoder* ae)
{
	return ae->layers;
}