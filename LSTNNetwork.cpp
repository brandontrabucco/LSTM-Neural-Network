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

vector<double> LSTMNetwork::classify(vector<double> input) {
	vector<double> output;
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		for (int i = 0; i < (blocks.size()); i++) {
			vector<double> activations = blocks[i].forward(input);
			for (int j = 0; j < activations.size(); j++)
				output.push_back(activations[i]);
		}
		return output;
	} else return output;
}

vector<double> LSTMNetwork::train(vector<double> input, vector<double> target) {
	vector<double> output;
	if (input.size() == inputSize && (blocks.size() * blocks[0].cells.size()) == target.size()) {
		// start forward pass
		for (int i = 0; i < (blocks.size()); i++) {
			vector<double> activations = blocks[i].forward(input);	// invalid read
			for (int j = 0; j < activations.size(); j++) {
				output.push_back(activations[i]);	// problem here, invalid read
			}
		}
		// start backward pass
		vector<double> error;
		for (int i = 0; i < output.size(); i++) {
			error.push_back(output[i] - target[i]);
		} output = error;
		for (int i = 0; i < (blocks.size()); i++) {
			// compute the activation
			vector<double> errorChunk((error.begin() + (i * blocks[i].cells.size())),
					(error.begin() + ((i + 1) * blocks[i].cells.size())));
			blocks[i].backward(errorChunk, learningRate);
		}
		learningRate *= decayRate;
		return output;
	} else {
		cout << "Target size mismatch" << endl;
		return output;
	}
}
