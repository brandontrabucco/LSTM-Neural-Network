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
	return (layers.size() == 0) ? (blocks.size() * blocks[0].cells.size()) : layers[layers.size() - 1].size();
}

void LSTMNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
}

vector<double> LSTMNetwork::classify(vector<double> input) {
	vector<double> output, connections = input;
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		for (int i = 0; i < (blocks.size()); i++) {
			vector<double> activations = blocks[i].forward(connections);
			for (int j = 0; j < activations.size(); j++)
				output.push_back(activations[j]);
		} connections = output;
		output.clear();
		for (int i = 0; i < layers.size(); i++) {
			vector<double> activations;
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations.push_back(layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output.push_back(activations[j]);
			} connections = activations;
		}
		return output;
	} else return output;
}

vector<double> LSTMNetwork::train(vector<double> input, vector<double> target) {
	vector<double> output, connections = input;
	if (input.size() == inputSize) {
		// start forward pass
		// calculate activations from bottom up
		for (int i = 0; i < (blocks.size()); i++) {
			vector<double> activations = blocks[i].forward(connections);
			for (int j = 0; j < activations.size(); j++)
				output.push_back(activations[j]);
		} connections = output;
		output.clear();
		for (int i = 0; i < layers.size(); i++) {
			vector<double> activations;
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations.push_back(layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output.push_back(activations[j]);
			} connections = activations;
		}
		// start backward pass
		vector<double> weightedError;
		for (int i = 0; i < output.size(); i++) {
			weightedError.push_back(output[i] - target[i]);
		} output = weightedError;
		for (int i = (layers.size() - 1); i >= 0; i--) {
			vector<double> errorSum(layers[i][0].weight.size(), 0.0);
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				vector<double> contribution = layers[i][j].backward(weightedError[j], learningRate);
				for (int k = 0; k < contribution.size(); k++) {
					errorSum[k] += contribution[k];
				}
			}
			weightedError = errorSum;
		} for (int i = 0; i < (blocks.size()); i++) {
			// compute the activation
			vector<double> errorChunk((weightedError.begin() + (i * blocks[i].cells.size())),
					(weightedError.begin() + ((i + 1) * blocks[i].cells.size())));
			blocks[i].backward(errorChunk, learningRate);
		}
		learningRate *= decayRate;
		return output;
	} else {
		cout << "Target size mismatch" << endl;
		return output;
	}
}
