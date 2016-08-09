/*
 * LSTMNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "LSTMNetwork.h"

LSTMNetwork::LSTMNetwork(int is, int b, int c, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
	for (int i = 0; i < b; i++) {
		blocks.push_back(MemoryBlock(c, is));
	}
}

LSTMNetwork::~LSTMNetwork() {
	// TODO Auto-generated destructor stub
}

int LSTMNetwork::getPreviousNeurons() {
	return (layers.size() == 0) ? (blocks.size() * blocks[0].nCells) : layers[layers.size() - 1].size();
}

void LSTMNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
}

vector<double> LSTMNetwork::classify(vector<double> input) {
	double *output = (double *)malloc(blocks.size() * blocks[0].nCells * sizeof(double)),
			*connections = (double *)malloc(sizeof(double) * input.size());
	copy(input.begin(), input.end(), connections);
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		#pragma omp parallel for
		for (int i = 0; i < (blocks.size()); i++) {
			double *activations = blocks[i].forward(connections);
			memcpy(&output[i * blocks[i].nCells], &activations[0], (sizeof(double) * blocks[i].nCells));
			free(activations);
		} connections = (double *)realloc(connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
		memcpy(connections, output, (sizeof(double) * blocks.size() * blocks[0].nCells));
		output = (double *)realloc(output, (sizeof(double) * layers[layers.size() - 1].size()));
		for (int i = 0; i < layers.size(); i++) {
			double *activations = (double *)malloc(sizeof(double) * layers[i].size());
			#pragma omp parallel for
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations[j] = (layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output[j] = (activations[j]);
			} connections = (double *)realloc(connections, (sizeof(double) * layers[i].size()));
			memcpy(connections, activations, (sizeof(double) * layers[i].size()));
			free(activations);
		} vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
		free(output);
		free(connections);
		return result;
	} else {
		cout << "Target size mismatch " << input.size() << ":" << inputSize << endl;
		return vector<double>();
	}
}

vector<double> LSTMNetwork::train(vector<double> input, vector<double> target) {
	double *output = (double *)malloc(blocks.size() * blocks[0].nCells * sizeof(double)),
			*connections = (double *)malloc(sizeof(double) * input.size());
	copy(input.begin(), input.end(), connections);
	if (input.size() == inputSize) {
		// start forward pass
		// calculate activations from bottom up
		#pragma omp parallel for
		for (int i = 0; i < (blocks.size()); i++) {
			double *activations = blocks[i].forward(connections);
			memcpy(&output[i * blocks[i].nCells], &activations[0], (sizeof(double) * blocks[i].nCells));
			free(activations);
		} connections = (double *)realloc(connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
		memcpy(connections, output, (sizeof(double) * blocks.size() * blocks[0].nCells));
		output = (double *)realloc(output, (sizeof(double) * layers[layers.size() - 1].size()));
		for (int i = 0; i < layers.size(); i++) {
			double *activations = (double *)malloc(sizeof(double) * layers[i].size());
			#pragma omp parallel for
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations[j] = (layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output[j] = (activations[j]);
			} connections = (double *)realloc(connections, (sizeof(double) * layers[i].size()));
			memcpy(connections, activations, (sizeof(double) * layers[i].size()));
			free(activations);
		} free(connections);
		// start backward pass
		double *weightedError = (double *)malloc((sizeof(double) * layers[layers.size() - 1].size()));
		#pragma omp parallel for
		for (int i = 0; i < layers[layers.size() - 1].size(); i++) {
			weightedError[i] = (output[i] - target[i]);
		} memcpy(output, weightedError, (sizeof(double) * layers[layers.size() - 1].size()));
		for (int i = (layers.size() - 1); i >= 0; i--) {
			double *errorSum = (double *)calloc(layers[i][0].connections, sizeof(double));
			#pragma omp parallel for
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				double *contribution = layers[i][j].backward(weightedError[j], learningRate);
				#pragma omp critical
				for (int k = 0; k < layers[i][0].connections; k++) {
					errorSum[k] += contribution[k];
				}
				free(contribution);
			} weightedError = (double *)realloc(weightedError, (sizeof(double) * layers[i][0].connections));
			memcpy(weightedError, errorSum, (sizeof(double) * layers[i][0].connections));
			free(errorSum);
		}
		#pragma omp parallel for
		for (int i = 0; i < (blocks.size()); i++) {
			// compute the activation
			double *errorChunk = (double *)malloc(sizeof(double) * blocks[i].nCells);
			memcpy(&errorChunk[0], &weightedError[(i * blocks[i].nCells)], (sizeof(double) * blocks[i].nCells));
			double *contribution = blocks[i].backward(errorChunk, learningRate);
			free(contribution);
			free(errorChunk);
		} learningRate *= decayRate;
		vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
		free(weightedError);
		free(output);
		return result;
	} else {
		cout << "Target size mismatch " << input.size() << ":" << inputSize << endl;
		return vector<double>();
	}
}
